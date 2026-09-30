"""Beat / downbeat ground truth for EGSet12 from its Guitar Pro tabs, and
(beat-tracker scoring, below) a benchmark for rhythm work. Measurement only.

EGSet12 ships a Guitar Pro 7/8 file per performance (a zip with
Content/score.gpif, XML): time signature per bar, tempo automations, and
every note's rhythm (note value, dots, tuplets). The JAMS annotations give
each note's real onset in the audio (within ~15ms) but no beats. This
script lines the two up:

  1. Parse the score into note onsets in quarter notes from the start
     (tied continuations aren't onsets; they add to the tied note's
     length) and bar starts.
  2. Match score notes to JAMS notes of the same pitch: first a global
     linear map (offset + seconds per quarter, the GP tempo as a start),
     then refit on the matches.
  3. Beats = the fitted grid at every quarter note, downbeats = at every
     bar start. All 12 performances were played to the GP tempo (fitted
     tempo within 0.3 bpm of it; strict-grid error vs the JAMS onsets
     median 4-20ms, p90 <= 40ms), so a strict grid is the ground truth. A
     local piecewise map through the matched notes is also reported, as a
     check: it isn't better (it follows strum spread and expressive
     timing), which is what a click-played performance looks like.

Usage (inside the worker container):
    python -m scripts.rhythm_benchmark truth [--show 01 05]
    python -m scripts.rhythm_benchmark eval

eval scores every tracker output under data/_rhythm/<tracker>/<tone>_<NN>.json
({"beats": [...], "downbeats": [...], optional "tempo_bpm"}; see
experiments/rhythm/track.py) against the truth, per tone:
  - beat F-measure, +-70ms (mir_eval), over the whole clip (the MIREX
    convention skips the first 5s; that score is kept in the JSON too);
  - downbeat F-measure, +-70ms;
  - tempo octave: 60 / median inter-beat interval vs the true tempo -
    right (within 8%), half, double, or other; and the same for a tracker's
    own global "tempo_bpm" if it reports one (librosa's is what the app
    shows today).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
import zipfile  # noqa: E402

import mir_eval  # noqa: E402
import numpy as np  # noqa: E402

from scripts import build_info  # noqa: E402
from scripts.egset12_benchmark import DATA_DIR, PERFORMANCES, TONES, load_truth  # noqa: E402

TRUTH_JSON = os.path.join(DATA_DIR, "rhythm_truth.json")
TRACKS_DIR = "/app/data/_rhythm"
TOLERANCE_S = 0.07
TEMPO_TOLERANCE = 0.08
NOTE_VALUE_QUARTERS = {"Whole": 4.0, "Half": 2.0, "Quarter": 1.0, "Eighth": 0.5, "16th": 0.25, "32nd": 0.125,
                       "64th": 0.0625}
MATCH_S = 0.12


# ------------------------------------------------------------ Guitar Pro


def parse_gp(path: str) -> dict:
    """Score from a .gp (GP7/8) file: bars (start and length in quarters,
    time signature), tempo automations, and note onsets (quarters from the
    start, sounding MIDI pitch, string 1 = high e ... 6 = low E, fret)."""
    root = ET.fromstring(zipfile.ZipFile(path).read("Content/score.gpif"))
    by_id = lambda tag: {e.get("id"): e for e in root.iter(tag) if e.get("id") is not None}  # noqa: E731
    bars, voices, beats, notes, rhythms = (by_id(t) for t in ("Bar", "Voice", "Beat", "Note", "Rhythm"))

    def rhythm_quarters(r) -> float:
        q = NOTE_VALUE_QUARTERS[r.findtext("NoteValue")]
        dot = r.find("AugmentationDot")
        if dot is not None:
            q *= {1: 1.5, 2: 1.75}[int(dot.get("count"))]
        tup = r.find("PrimaryTuplet")
        if tup is not None:
            q *= int(tup.get("den")) / int(tup.get("num"))
        return q

    def prop(note, name, child):
        for p in note.iter("Property"):
            if p.get("name") == name:
                return p.findtext(child)
        return None

    tempos = [{"bar": int(a.findtext("Bar")), "position": float(a.findtext("Position")),
               "bpm": float(a.findtext("Value").split()[0]), "unit": a.findtext("Value").split()[1]}
              for a in root.iter("Automation") if a.findtext("Type") == "Tempo"]
    bar_list, out_notes, start = [], [], 0.0
    tied: dict[tuple, dict] = {}  # (voice slot, string) -> the note a tie continues
    for i, mb in enumerate(root.find("MasterBars")):
        num, den = (int(v) for v in mb.findtext("Time").split("/"))
        length = num * 4.0 / den
        bar_list.append({"index": i, "start_q": start, "length_q": length, "time_signature": f"{num}/{den}"})
        bar = bars[mb.findtext("Bars").split()[0]]  # first (only) track
        for slot, voice_id in enumerate(bar.findtext("Voices").split()):
            if voice_id == "-1":
                continue
            pos = start
            for beat_id in (voices[voice_id].findtext("Beats") or "").split():
                beat = beats[beat_id]
                rhythm = rhythms[beat.find("Rhythm").get("ref")]
                dur = rhythm_quarters(rhythm)
                for note_id in (beat.findtext("Notes") or "").split():
                    note = notes[note_id]
                    string = 6 - int(prop(note, "String", "String"))
                    tie = note.find("Tie")
                    if tie is not None and tie.get("destination") == "true" and (slot, string) in tied:
                        # Continuation of a tied note: not an onset, but part of its length.
                        tied[(slot, string)]["dur_q"] += dur
                        continue
                    out_notes.append({"q": round(pos, 6), "dur_q": dur, "midi": int(prop(note, "Midi", "Number")),
                                      "string": string, "fret": int(prop(note, "Fret", "Fret")),
                                      "tuplet": rhythm.find("PrimaryTuplet") is not None})
                    tied[(slot, string)] = out_notes[-1]
                pos += dur
        start += length
    return {"bars": bar_list, "tempos": tempos, "notes": out_notes, "length_q": start}


# ------------------------------------------------------------- alignment


def _match(score: list[dict], audio: list[dict], offset: float, spq: float, window: float) -> list[tuple]:
    """One-to-one (score note, audio note) pairs of the same pitch whose
    onsets land within `window` under t = offset + q * spq; closest first."""
    pairs = sorted((abs(a["onset"] - (offset + s["q"] * spq)), i, j)
                   for i, s in enumerate(score) for j, a in enumerate(audio)
                   if s["midi"] == a["midi"] and abs(a["onset"] - (offset + s["q"] * spq)) <= window)
    used_s, used_a, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_s and j not in used_a:
            used_s.add(i)
            used_a.add(j)
            out.append((i, j))
    return out


def align(score: dict, audio_notes: list[dict]) -> dict:
    notes = score["notes"]
    spq0 = 60.0 / score["tempos"][0]["bpm"]
    # Offset: the one under which most score notes find an audio note.
    first = audio_notes[0]["onset"]
    best = max((len(_match(notes, audio_notes, off, spq0, 0.08)), off)
               for off in np.arange(first - 4.0, first + 1.0, 0.01))
    offset, spq = best[1], spq0
    for window in (MATCH_S, 0.08):  # refit tempo and offset on the matches
        pairs = _match(notes, audio_notes, offset, spq, window)
        q = np.array([notes[i]["q"] for i, _ in pairs])
        t = np.array([audio_notes[j]["onset"] for _, j in pairs])
        spq, offset = np.polyfit(q, t, 1)
    pairs = _match(notes, audio_notes, offset, spq, MATCH_S)
    q = np.array([notes[i]["q"] for i, _ in pairs])
    t = np.array([audio_notes[j]["onset"] for _, j in pairs])
    linear_resid = t - (offset + q * spq)

    # Local map: one anchor per score position (median onset of its matched
    # notes, e.g. a strum), made monotonic, interpolated linearly and
    # extrapolated at the fitted tempo beyond the first/last anchor.
    anchors: dict[float, list[float]] = {}
    for qi, ti in zip(q, t):
        anchors.setdefault(float(qi), []).append(float(ti))
    aq = np.array(sorted(anchors))
    at = np.maximum.accumulate(np.array([np.median(anchors[k]) for k in aq]))

    def local(x):
        x = np.asarray(x, dtype=float)
        y = np.interp(x, aq, at)
        y = np.where(x < aq[0], at[0] + (x - aq[0]) * spq, y)
        return np.where(x > aq[-1], at[-1] + (x - aq[-1]) * spq, y)

    # Leave-one-out check of the local map: predict each anchor from its neighbours.
    loo = [float(at[k] - np.interp(aq[k], np.delete(aq, k), np.delete(at, k))) for k in range(1, len(aq) - 1)]
    return {
        "offset": float(offset), "spq": float(spq), "fitted_bpm": 60.0 / float(spq),
        "matched_score": len(pairs), "score_onsets": len(notes), "audio_notes": len(audio_notes),
        "linear_resid_ms": {"median": float(np.median(np.abs(linear_resid)) * 1000),
                            "p90": float(np.percentile(np.abs(linear_resid), 90) * 1000),
                            "max": float(np.max(np.abs(linear_resid)) * 1000)},
        "local_loo_ms": {"median": float(np.median(np.abs(loo)) * 1000),
                         "p90": float(np.percentile(np.abs(loo), 90) * 1000)} if loo else None,
        "local": local,
        "pairs": pairs,
    }


def build_truth(show: list[str]) -> dict:
    truth = {}
    for p in PERFORMANCES:
        score = parse_gp(os.path.join(DATA_DIR, f"{p}.gp"))
        audio_notes, jams_tempo = load_truth(p)
        a = align(score, audio_notes)
        duration = max(n["end"] for n in audio_notes)
        beat_q = np.arange(0.0, score["length_q"] + 1e-9, 1.0)
        bar_q = np.array([b["start_q"] for b in score["bars"]])
        grid = lambda q: a["offset"] + np.asarray(q) * a["spq"]  # noqa: E731
        beats = [float(x) for x in grid(beat_q) if 0.0 <= x <= duration + 0.5]
        downbeats = [float(x) for x in grid(bar_q) if 0.0 <= x <= duration + 0.5]
        # Inter-beat intervals of the local map: how far the player strays from the grid tempo.
        ibi = np.diff(a["local"](beat_q))
        # Score notes with their rhythm, linked to the JAMS note they matched
        # (for scoring quantization); unmatched score notes have none.
        jams_of = dict(a["pairs"])
        score_notes = [{**{k: n[k] for k in ("q", "dur_q", "midi", "tuplet")},
                        **({"onset": round(audio_notes[jams_of[i]]["onset"], 4),
                            "end": round(audio_notes[jams_of[i]]["end"], 4)} if i in jams_of else {})}
                       for i, n in enumerate(score["notes"])]
        truth[p] = {
            "gp_bpm": score["tempos"][0]["bpm"], "jams_bpm": jams_tempo,
            "time_signatures": sorted({b["time_signature"] for b in score["bars"]}),
            "bars": len(score["bars"]), "fitted_bpm": round(a["fitted_bpm"], 2),
            "performed_bpm_p10_p90": [round(60 / np.percentile(ibi, 90), 1), round(60 / np.percentile(ibi, 10), 1)],
            "offset_s": round(a["offset"], 3), "matched": f"{a['matched_score']}/{a['score_onsets']} score onsets "
                                                           f"({a['audio_notes']} JAMS notes)",
            "linear_resid_ms": {k: round(v, 1) for k, v in a["linear_resid_ms"].items()},
            "local_loo_ms": {k: round(v, 1) for k, v in a["local_loo_ms"].items()} if a["local_loo_ms"] else None,
            "beats": [round(b, 4) for b in beats], "downbeats": [round(d, 4) for d in downbeats],
            # The grid itself: time = offset + quarter index * spq.
            "grid": {"offset": a["offset"], "spq": a["spq"]},
            "notes": score_notes,
        }
        if p in show:
            print(f"\n{p}: GP {truth[p]['gp_bpm']:g} bpm (JAMS {jams_tempo}), {', '.join(truth[p]['time_signatures'])}, "
                  f"{truth[p]['bars']} bars; fitted {truth[p]['fitted_bpm']} bpm, first bar at {a['offset']:.3f}s")
            print(f"    matched {truth[p]['matched']}")
            print(f"    strict-tempo grid vs JAMS onsets: |error| median {truth[p]['linear_resid_ms']['median']}ms, "
                  f"p90 {truth[p]['linear_resid_ms']['p90']}ms, max {truth[p]['linear_resid_ms']['max']}ms")
            if truth[p]["local_loo_ms"]:
                print(f"    local map, leave-one-out: median {truth[p]['local_loo_ms']['median']}ms, "
                      f"p90 {truth[p]['local_loo_ms']['p90']}ms; performed tempo p10-p90 "
                      f"{truth[p]['performed_bpm_p10_p90']} bpm")
    with open(TRUTH_JSON, "w") as f:
        json.dump(truth, f, indent=1)
    return truth


# ------------------------------------------------------------------ eval


def tempo_octave(estimate: float | None, true_bpm: float) -> str:
    if not estimate:
        return "none"
    ratio = estimate / true_bpm
    for name, target in (("right", 1.0), ("half", 0.5), ("double", 2.0)):
        if abs(ratio / target - 1) <= TEMPO_TOLERANCE:
            return name
    return "other"


def evaluate(truth: dict) -> dict:
    result = {}
    for tracker in sorted(os.listdir(TRACKS_DIR)):
        for tone in TONES:
            rows = []
            for p in PERFORMANCES:
                path = os.path.join(TRACKS_DIR, tracker, f"{tone}_{p}.json")
                if not os.path.exists(path):
                    continue
                est = json.load(open(path))
                ref_b, ref_d = np.array(truth[p]["beats"]), np.array(truth[p]["downbeats"])
                est_b, est_d = np.array(est["beats"]), np.array(est["downbeats"])
                ibi = np.diff(est_b)
                beat_bpm = 60.0 / float(np.median(ibi)) if len(ibi) else None
                rows.append({
                    "performance": p,
                    "beat_f": mir_eval.beat.f_measure(ref_b, est_b, TOLERANCE_S),
                    "beat_f_mirex": mir_eval.beat.f_measure(mir_eval.beat.trim_beats(ref_b),
                                                            mir_eval.beat.trim_beats(est_b), TOLERANCE_S),
                    # Diagnostic: the best F over the tracker's beats as they are, doubled
                    # (midpoints added) or halved (either phase), i.e. timing accuracy
                    # with the half/double choice taken out.
                    "beat_f_any_octave": max(mir_eval.beat.f_measure(ref_b, v, TOLERANCE_S) for v in (
                        est_b, np.sort(np.concatenate([est_b, est_b[:-1] + ibi / 2])) if len(ibi) else est_b,
                        est_b[0::2], est_b[1::2])),
                    "downbeat_f": mir_eval.beat.f_measure(ref_d, est_d, TOLERANCE_S),
                    "true_bpm": truth[p]["fitted_bpm"], "beat_bpm": beat_bpm,
                    "tempo_octave": tempo_octave(beat_bpm, truth[p]["fitted_bpm"]),
                    "global_bpm": est.get("tempo_bpm"),
                    "global_octave": tempo_octave(est.get("tempo_bpm"), truth[p]["fitted_bpm"])
                    if "tempo_bpm" in est else None,
                    "seconds_per_second": est.get("seconds", 0) / est.get("audio_seconds", 1),
                })
            if rows:
                result[f"{tracker}|{tone}"] = rows
    return result


def print_eval(result: dict) -> None:
    def octaves(rows, key):
        counts = {}
        for r in rows:
            counts[r[key]] = counts.get(r[key], 0) + 1
        return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))

    print("tracker    tone      beat F  (MIREX)  any-octave  downbeat F   tempo octave (from beats)        s per s")
    for key, rows in result.items():
        tracker, tone = key.split("|")
        mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
        print(f"{tracker:<10} {tone:<9} {mean('beat_f'):6.1%}  ({mean('beat_f_mirex'):5.1%})  {mean('beat_f_any_octave'):9.1%}"
              f"  {mean('downbeat_f'):9.1%}"
              f"   {octaves(rows, 'tempo_octave'):<32} {mean('seconds_per_second'):.3f}")
        if rows[0]["global_octave"] is not None:
            print(f"{'':<10} {'':<9} global tempo_bpm (what the app shows): {octaves(rows, 'global_octave')}")
    print("\nper performance, clean: beat F / downbeat F / tempo from beats (true)")
    for key, rows in result.items():
        if key.endswith("|clean"):
            print(f"  {key.split('|')[0]:<10} " + "  ".join(
                f"{r['performance']}: {r['beat_f']:.2f}/{r['downbeat_f']:.2f}/{(r['beat_bpm'] or 0):.0f}({r['true_bpm']:.0f})"
                for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["truth", "eval"])
    parser.add_argument("--show", nargs="*", default=[])
    build_info.add_args(parser)
    args = parser.parse_args()
    build_info.guard(args.allow_stale)
    if args.command == "eval":
        result = evaluate(json.load(open(TRUTH_JSON)))
        json.dump(result, open(os.path.join(TRACKS_DIR, "eval.json"), "w"), indent=1)
        print_eval(result)
        return
    truth = build_truth(args.show)
    print("\nperf  GP bpm  JAMS  fitted  sig       matched                          grid |err| med/p90  local LOO med/p90")
    for p, t in truth.items():
        loo = t["local_loo_ms"] or {"median": float("nan"), "p90": float("nan")}
        print(f"  {p}  {t['gp_bpm']:>6g}  {t['jams_bpm']:>4}  {t['fitted_bpm']:>6}  {'/'.join(t['time_signatures']):<8}  "
              f"{t['matched']:<32} {t['linear_resid_ms']['median']:>5}/{t['linear_resid_ms']['p90']:<6}   "
              f"{loo['median']:>5}/{loo['p90']}")


if __name__ == "__main__":
    main()
