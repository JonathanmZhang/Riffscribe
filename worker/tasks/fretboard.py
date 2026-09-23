import logging

import pretty_midi

from tasks.celery_app import app
from tasks.storage import get_job, update_job

logger = logging.getLogger(__name__)

STANDARD_TUNING = {6: "E2", 5: "A2", 4: "D3", 3: "G3", 2: "B3", 1: "E4"}
OPEN_STRING_MIDI = {string: pretty_midi.note_name_to_number(name) for string, name in STANDARD_TUNING.items()}
MAX_FRET = 20

# Notes whose start_time falls within this window of each other are treated
# as one chord (simultaneous), per spec 3.5. Note timestamps come from ML
# inference so exact equality can't be relied on.
CHORD_ONSET_TOLERANCE_SECONDS = 0.05

DEFAULT_TEMPO_BPM = 120.0  # placeholder: transcribe() doesn't estimate tempo yet


def _candidates_for_pitch(pitch_midi: int) -> list[tuple[int, int]]:
    """Every valid (string, fret) pair in 0-20 that produces this MIDI pitch."""
    candidates = []
    for string, open_midi in OPEN_STRING_MIDI.items():
        fret = pitch_midi - open_midi
        if 0 <= fret <= MAX_FRET:
            candidates.append((string, fret))
    return candidates


def _position_cost(prev: tuple[int, int], curr: tuple[int, int]) -> float:
    """Cost of moving from one fretboard position to the next, per spec 3.5."""
    s1, f1 = prev
    s2, f2 = curr
    cost = abs(f1 - f2)
    if f2 == 0:
        cost += 2
    if abs(f1 - f2) > 4:
        cost += 3
    if s1 == s2:
        cost += 1
    return cost


def _transition_cost(prev_positions: list[tuple[int, int]], curr_positions: list[tuple[int, int]]) -> float:
    """Generalizes the single-position cost formula to chord steps: the
    average pairwise cost across every note in the previous step and every
    note in the current step. For the monophonic case (the common one) each
    list has exactly one element, so this reduces to the spec's formula
    exactly.
    """
    pairwise = [_position_cost(prev, curr) for prev in prev_positions for curr in curr_positions]
    return sum(pairwise) / len(pairwise)


def _candidate_fret_sum(positions: list[tuple[int, int]]) -> int:
    return sum(fret for _, fret in positions)


def _voice_chord(pitches_midi: list[int]) -> list[tuple[int, int]]:
    """v1 chord voicing: assign each note the lowest available fret on an
    unused string (hand-shape realism is a stretch goal per spec 3.5).
    """
    order = sorted(range(len(pitches_midi)), key=lambda i: -pitches_midi[i])
    used_strings: set[int] = set()
    assignment: list[tuple[int, int] | None] = [None] * len(pitches_midi)

    for i in order:
        candidates = sorted(_candidates_for_pitch(pitches_midi[i]), key=lambda c: c[1])
        for string, fret in candidates:
            if string not in used_strings:
                assignment[i] = (string, fret)
                used_strings.add(string)
                break
        else:
            raise ValueError(
                f"no available string/fret for MIDI pitch {pitches_midi[i]} "
                f"within a {len(pitches_midi)}-note chord"
            )

    return assignment  # type: ignore[return-value]


def _group_into_steps(notes: list[dict]) -> list[list[dict]]:
    """Groups simultaneous notes (chords) together; each group is processed
    as one DP step.
    """
    steps: list[list[dict]] = []
    current_group: list[dict] = []
    group_start = None

    for note in sorted(notes, key=lambda n: n["start_time"]):
        if current_group and note["start_time"] - group_start > CHORD_ONSET_TOLERANCE_SECONDS:
            steps.append(current_group)
            current_group = []
            group_start = None
        if group_start is None:
            group_start = note["start_time"]
        current_group.append(note)

    if current_group:
        steps.append(current_group)

    return steps


def map_notes_to_positions(notes: list[dict]) -> list[dict]:
    """Maps note events onto guitar string/fret positions via dynamic
    programming, per spec 3.5. dp[i][c] = minimum cumulative cost to reach
    candidate c at step i; backpointers reconstruct the globally optimal
    path once the full table is built (Viterbi-style), not a greedy
    nearest-position choice.

    The cost formula in spec 3.5 has no bias toward absolute neck position
    (it only penalizes relative movement, same-string reuse, and open
    strings), so distinct regions of the neck can tie exactly on total cost.
    Each dp value is a (transition_cost, fret_sum) tuple compared
    lexicographically, so ties are broken in favor of the lowest-fret path
    among all cost-optimal paths, without changing the cost formula itself.
    """
    if not notes:
        return []

    steps = _group_into_steps(notes)

    # step_candidates[i] is a list of candidate positions for step i; a
    # monophonic step gets every valid (string, fret) for its pitch (real
    # DP choice), a chord step gets exactly one pre-voiced combination.
    step_candidates: list[list[list[tuple[int, int]]]] = []
    for step_notes in steps:
        pitches_midi = [pretty_midi.note_name_to_number(n["pitch"]) for n in step_notes]
        if len(step_notes) == 1:
            singles = _candidates_for_pitch(pitches_midi[0])
            if not singles:
                raise ValueError(
                    f"pitch {step_notes[0]['pitch']} has no valid fret 0-{MAX_FRET} "
                    "position in standard tuning"
                )
            candidates = [[c] for c in singles]
        else:
            candidates = [_voice_chord(pitches_midi)]
        step_candidates.append(candidates)

    dp: list[list[tuple[float, int]]] = [
        [(0.0, _candidate_fret_sum(c)) for c in step_candidates[0]]
    ]
    backptr: list[list[int | None]] = [[None] * len(step_candidates[0])]

    for i in range(1, len(step_candidates)):
        prev_candidates = step_candidates[i - 1]
        curr_candidates = step_candidates[i]
        dp_row: list[tuple[float, int]] = []
        bp_row: list[int | None] = []

        for curr in curr_candidates:
            curr_fret_sum = _candidate_fret_sum(curr)
            best_value = None
            best_prev_idx = None
            for prev_idx, prev in enumerate(prev_candidates):
                prev_cost, prev_fret_sum = dp[i - 1][prev_idx]
                value = (prev_cost + _transition_cost(prev, curr), prev_fret_sum + curr_fret_sum)
                if best_value is None or value < best_value:
                    best_value = value
                    best_prev_idx = prev_idx
            dp_row.append(best_value)
            bp_row.append(best_prev_idx)

        dp.append(dp_row)
        backptr.append(bp_row)

    last_row = dp[-1]
    chosen_indices: list[int] = [0] * len(step_candidates)
    chosen_indices[-1] = min(range(len(last_row)), key=lambda i: last_row[i])
    for i in range(len(step_candidates) - 1, 0, -1):
        chosen_indices[i - 1] = backptr[i][chosen_indices[i]]

    result_notes: list[dict] = []
    for step_notes, candidates, idx in zip(steps, step_candidates, chosen_indices):
        chosen_positions = candidates[idx]
        for note, (string, fret) in zip(step_notes, chosen_positions):
            result_notes.append(
                {
                    "string": string,
                    "fret": fret,
                    "start_time": note["start_time"],
                    "end_time": note["end_time"],
                    "pitch": note["pitch"],
                }
            )

    return result_notes


@app.task(name="map_fretboard", soft_time_limit=120)
def map_fretboard(job_id: str) -> str:
    try:
        update_job(job_id, status="processing")

        job = get_job(job_id)
        raw_note_events = job.get("raw_note_events")
        if raw_note_events is None:
            raise ValueError(f"job {job_id} has no raw_note_events; transcribe must run first")

        logger.info("map_fretboard: job %s mapping %d note(s)", job_id, len(raw_note_events))

        mapped_notes = map_notes_to_positions(raw_note_events)
        duration_seconds = max((note["end_time"] for note in mapped_notes), default=0.0)

        result = {
            "job_id": job_id,
            "duration_seconds": duration_seconds,
            "tempo_bpm": DEFAULT_TEMPO_BPM,
            "notes": mapped_notes,
        }

        update_job(job_id, status="done", result=result)
        logger.info("map_fretboard: job %s done with %d mapped note(s)", job_id, len(mapped_notes))
    except Exception as exc:
        logger.exception("map_fretboard failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"map_fretboard failed: {exc}")
        raise
    return job_id
