"""Bar lines: choose the bar phase (which beat is the "1") from chord
changes, optionally with low-note onsets, on top of beat_this's beats.
Scores downbeat F (+-70ms) against the Guitar Pro truth, all three tones.
Runs in the worker image after track.py (it reads beat_this's output):
    python /rhythm/bars.py

Every variant keeps beat_this's beats and only chooses the phase:
downbeats = beats[k::M], one k for the whole clip, or ("local") k chosen
per beat from the evidence within +-2 bars. The bar length M is either
beat_this's (the most common beat count between its downbeats; 4 if it
has fewer than two) or assumed 4. Evidence, per beat, each normalized to
sum 1 over the clip:
  chords     BTC chord changes (tasks/chords.recognize_chords on the
             pipeline's audio): each segment start, assigned to its nearest
             beat (within 0.35 beat), weighted by the new chord's length in
             beats (capped at a bar), so one-frame flickers count little.
  bass       onsets of the pipeline's kept notes at or below E3 (MIDI 52),
             weighted by amplitude, assigned to the nearest beat (within
             0.25 beat).
  beat_this  1 on the beats beat_this marked as downbeats.
Weights are all 1: nothing is tuned on these 12 performances.

Variants: beat_this as it is (irregular: it can change phase mid-clip), its
phase made regular, chords, chords + bass, chords + beat_this, chords + bass
+ beat_this, an oracle (the one phase that scores best against the truth -
the ceiling for any one-phase choice on these beats), and local versions.
Result (README): nothing beats beat_this's own downbeats, not even the
one-phase oracle; chord changes are the weakest evidence.
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, "/app")

import mir_eval  # noqa: E402
import numpy as np  # noqa: E402

from scripts import build_info  # noqa: E402
from scripts.chord_stages import PIPELINE, detect_file, run_stages  # noqa: E402
from scripts.egset12_benchmark import PERFORMANCES, TONES, audio_path  # noqa: E402
from scripts.rhythm_benchmark import TOLERANCE_S, TRACKS_DIR, TRUTH_JSON  # noqa: E402
from tasks.audio_io import normalize_to_wav  # noqa: E402

EVIDENCE_DIR = os.path.join(TRACKS_DIR, "_evidence")
BASS_MAX_MIDI = 52


def evidence(tone: str, p: str) -> dict:
    """Chord segments and low-note onsets for one recording, cached."""
    path = os.path.join(EVIDENCE_DIR, f"{tone}_{p}.json")
    if os.path.exists(path):
        return json.load(open(path))
    from tasks.chords import recognize_chords

    with tempfile.TemporaryDirectory() as workdir:
        chords = recognize_chords(normalize_to_wav(audio_path(p, tone), os.path.join(workdir, "n.wav")))
        events, activations, _ = detect_file(audio_path(p, tone), workdir)
    kept = run_stages(events, PIPELINE, activations)["kept"]
    bass = [[round(n["start_time"], 4), round(n["amplitude"], 3)] for n in kept
            if n["midi"] <= BASS_MAX_MIDI]
    out = {"chords": chords, "bass": bass}
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    json.dump(out, open(path, "w"))
    return out


def beats_per_bar(beats: np.ndarray, downbeats: np.ndarray) -> int:
    if len(downbeats) < 2:
        return 4
    idx = [int(np.argmin(np.abs(beats - d))) for d in downbeats]
    counts = [b - a for a, b in zip(idx, idx[1:]) if b > a]
    return int(np.bincount(counts).argmax()) if counts else 4


def assign(beats: np.ndarray, times, weights, window: float) -> np.ndarray:
    """Sum of weights per beat, each time going to its nearest beat if
    within `window` beats of it."""
    out = np.zeros(len(beats))
    if len(beats) < 2:
        return out
    ibi = float(np.median(np.diff(beats)))
    for t, w in zip(times, weights):
        i = int(np.argmin(np.abs(beats - t)))
        if abs(beats[i] - t) <= window * ibi:
            out[i] += w
    return out


def normalized(x: np.ndarray) -> np.ndarray:
    return x / x.sum() if x.sum() > 0 else x


def phase_scores(per_beat: np.ndarray, m: int) -> np.ndarray:
    return np.array([per_beat[k::m].sum() for k in range(m)])


def pick_local(beats: np.ndarray, per_beat: np.ndarray, m: int, bars: int = 2) -> np.ndarray:
    """Downbeats with the phase chosen per beat from the evidence within
    +-`bars` bars of it, so one missed or extra beat only shifts the phase
    locally. A beat is a downbeat when it's the chosen phase there."""
    out = []
    for i in range(len(beats)):
        lo, hi = max(0, i - bars * m), min(len(beats), i + bars * m + 1)
        scores = [sum(per_beat[j] for j in range(lo, hi) if j % m == k) for k in range(m)]
        if i % m == int(np.argmax(scores)):
            out.append(beats[i])
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    build_info.add_args(parser)
    build_info.guard(parser.parse_args().allow_stale)
    truth = json.load(open(TRUTH_JSON))
    phases = ["beat_this regular", "chords", "chords+bass", "chords+beat_this", "chords+bass+beat_this",
              "oracle phase", "local beat_this regular", "local chords", "local chords+bass+beat_this"]
    variants = ["beat_this"] + [f"{meter}: {ph}" for meter in ("M from beat_this", "M=4") for ph in phases]
    rows = {}
    for tone in TONES:
        for p in PERFORMANCES:
            bt = json.load(open(os.path.join(TRACKS_DIR, "beat_this", f"{tone}_{p}.json")))
            beats, downbeats = np.array(bt["beats"]), np.array(bt["downbeats"])
            ref = np.array(truth[p]["downbeats"])
            ev = evidence(tone, p)
            ibi = float(np.median(np.diff(beats)))
            f = lambda est: mir_eval.beat.f_measure(ref, np.asarray(est), TOLERANCE_S)  # noqa: E731
            m_bt = beats_per_bar(beats, downbeats)
            row = {"m": m_bt, "beat_bpm": 60 / ibi, "beat_this": f(downbeats)}
            for meter, m in (("M from beat_this", m_bt), ("M=4", 4)):
                chord_w = [min((c["end"] - c["start"]) / ibi, m) for c in ev["chords"]]
                chords = normalized(assign(beats, [c["start"] for c in ev["chords"]], chord_w, 0.35))
                bass = normalized(assign(beats, [b[0] for b in ev["bass"]], [b[1] for b in ev["bass"]], 0.25))
                prior = normalized(assign(beats, downbeats, np.ones(len(downbeats)), 0.1))
                pick = lambda ev_: beats[int(np.argmax(phase_scores(ev_, m)))::m]  # noqa: E731
                row.update({
                    f"{meter}: beat_this regular": f(pick(prior)),
                    f"{meter}: chords": f(pick(chords)),
                    f"{meter}: chords+bass": f(pick(chords + bass)),
                    f"{meter}: chords+beat_this": f(pick(chords + prior)),
                    f"{meter}: chords+bass+beat_this": f(pick(chords + bass + prior)),
                    f"{meter}: oracle phase": max(f(beats[k::m]) for k in range(m)),
                    f"{meter}: local beat_this regular": f(pick_local(beats, prior, m)),
                    f"{meter}: local chords": f(pick_local(beats, chords, m)),
                    f"{meter}: local chords+bass+beat_this": f(pick_local(beats, chords + bass + prior, m)),
                })
            rows[tone, p] = row
    os.makedirs(TRACKS_DIR, exist_ok=True)
    json.dump({f"{t}|{p}": r for (t, p), r in rows.items()}, open(os.path.join(TRACKS_DIR, "bars.json"), "w"),
              indent=1)

    print("downbeat F (+-70ms), mean over 12 performances")
    print(f"{'variant':<42}" + "".join(f"{t:>10}" for t in TONES))
    for v in variants:
        print(f"{v:<42}" + "".join(f"{np.mean([rows[t, p][v] for p in PERFORMANCES]):>10.1%}" for t in TONES))
    short = ["beat_this", "M from beat_this: oracle phase", "M=4: beat_this regular", "M=4: chords",
             "M=4: chords+bass+beat_this", "M=4: oracle phase"]
    for tone in TONES:
        print(f"\nper performance, {tone}: beat_this's beats per bar (tracker bpm) | " + " | ".join(short))
        for p in PERFORMANCES:
            r = rows[tone, p]
            print(f"  {p}: M={r['m']} ({r['beat_bpm']:5.1f}) | " + " | ".join(f"{r[v]:.2f}" for v in short))


if __name__ == "__main__":
    main()
