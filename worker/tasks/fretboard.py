import logging
from itertools import combinations

import pretty_midi

from tasks.celery_app import app
from tasks.storage import get_job, update_job

logger = logging.getLogger(__name__)

STANDARD_TUNING = {6: "E2", 5: "A2", 4: "D3", 3: "G3", 2: "B3", 1: "E4"}
OPEN_STRING_MIDI = {string: pretty_midi.note_name_to_number(name) for string, name in STANDARD_TUNING.items()}
MAX_FRET = 20

# Notes whose start_time falls within this window of each other are treated
# as one chord (simultaneous), per spec 3.5. Note timestamps come from ML
# inference so exact equality can't be relied on. Must match
# STEP_TOLERANCE_SECONDS in the frontend's TabViewer.tsx. 150ms is an
# empirically-set value based on real strum testing, not a proven optimum -
# same status as the cost-function constants.
CHORD_ONSET_TOLERANCE_SECONDS = 0.15


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
        cost -= 2  # open-string bonus: reduces cost
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


def _voice_chord_greedy(pitches_midi: list[int]) -> list[tuple[int, int]] | None:
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
            return None

    return assignment  # type: ignore[return-value]


def _voice_chord_search(pitches_midi: list[int]) -> list[tuple[int, int]] | None:
    """Exhaustive fallback: lowest total-fret assignment onto distinct
    strings, or None if none exists. Only reached when the greedy pass fails,
    to tell "greedy took a string another note needed" (a voicing exists)
    apart from "genuinely impossible" (e.g. two notes that are only
    playable on the same string).
    """
    candidate_lists = [_candidates_for_pitch(p) for p in pitches_midi]
    best: tuple[int, list[tuple[int, int]]] | None = None

    def recurse(i: int, used: set[int], chosen: list[tuple[int, int]], fret_sum: int) -> None:
        nonlocal best
        if best is not None and fret_sum >= best[0]:
            return
        if i == len(candidate_lists):
            best = (fret_sum, list(chosen))
            return
        for string, fret in sorted(candidate_lists[i], key=lambda c: c[1]):
            if string in used:
                continue
            used.add(string)
            chosen.append((string, fret))
            recurse(i + 1, used, chosen, fret_sum + fret)
            chosen.pop()
            used.discard(string)

    recurse(0, set(), [], 0)
    return best[1] if best else None


def _voice_chord(pitches_midi: list[int]) -> list[tuple[int, int]] | None:
    """v1 chord voicing: assign each note the lowest available fret on an
    unused string (hand-shape realism is a stretch goal per spec 3.5).
    Returns None if the notes can't be placed on distinct strings.
    """
    return _voice_chord_greedy(pitches_midi) or _voice_chord_search(pitches_midi)


def _make_voiceable(step_notes: list[dict]) -> tuple[list[dict], list[tuple[int, int]] | None]:
    """For a chord that can't be voiced on distinct strings, drops notes
    (unplayable-range notes first, then the lowest-amplitude conflicting set) until it can, per
    spec 3.7: an unplayable chord still produces a result, not a failed job.
    Returns the kept notes and their voicing (None for a single remaining
    note, which the caller handles as a monophonic step).
    """
    notes = list(step_notes)

    while len(notes) > 1:
        pitches = [pretty_midi.note_name_to_number(n["pitch"]) for n in notes]
        voicing = _voice_chord(pitches)
        if voicing is not None:
            return notes, voicing

        out_of_range = [i for i, p in enumerate(pitches) if not _candidates_for_pitch(p)]
        if out_of_range:
            victims = [out_of_range[0]]
            reason = f"outside the playable range (frets 0-{MAX_FRET}, standard tuning)"
        else:
            victims = _cheapest_drop_set(notes, pitches)
            reason = (
                "the chord cannot be voiced on distinct strings; dropped the "
                "lowest-total-amplitude set of notes that resolves the conflict"
            )

        chord_size = len(notes)
        for i in sorted(victims, reverse=True):
            dropped = notes.pop(i)
            logger.warning(
                "map_fretboard: dropped %s (start %.2fs, amplitude %s) from a %d-note chord: %s",
                dropped["pitch"],
                dropped["start_time"],
                dropped.get("amplitude", "n/a"),
                chord_size,
                reason,
            )

    return notes, None


def _cheapest_drop_set(notes: list[dict], pitches: list[int]) -> list[int]:
    """Smallest set of notes to drop that makes the chord voiceable, breaking
    ties by lowest total amplitude. Dropping notes one at a time by amplitude
    alone can remove notes that aren't part of the conflict (e.g. two notes
    that both need the low E string) while leaving the conflict in place.
    """
    for size in range(1, len(notes)):
        viable = []
        for drop in combinations(range(len(notes)), size):
            kept = [p for i, p in enumerate(pitches) if i not in drop]
            if _voice_chord(kept) is not None:
                viable.append(drop)
        if viable:
            best = min(viable, key=lambda drop: sum(notes[i].get("amplitude", 0.0) for i in drop))
            return list(best)
    return list(range(len(notes) - 1))


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
    (it only penalizes relative movement and same-string reuse, and rewards
    open strings), so distinct regions of the neck can tie exactly on total
    cost.
    Each dp value is a (transition_cost, fret_sum) tuple compared
    lexicographically, so ties are broken in favor of the lowest-fret path
    among all cost-optimal paths, without changing the cost formula itself.
    """
    return map_notes_with_steps(notes)[0]


def map_notes_with_steps(notes: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """map_notes_to_positions, also returning the groups (tab columns) the
    notes were placed in, including notes later dropped as unplayable. For
    the measurement scripts, so they see exactly the grouping the mapper
    used; the pipeline calls map_notes_to_positions.
    """
    if not notes:
        return [], []

    # step_candidates[i] is a list of candidate positions for step i; a
    # monophonic step gets every valid (string, fret) for its pitch (real
    # DP choice), a chord step gets exactly one pre-voiced combination.
    # Chords that can't be voiced are reduced first (notes dropped), so
    # `steps` holds only the notes that actually get placed.
    groups: list[list[dict]] = []
    steps: list[list[dict]] = []
    step_candidates: list[list[list[tuple[int, int]]]] = []
    for raw_step in _group_into_steps(notes):
        groups.append(raw_step)
        step_notes, voicing = _make_voiceable(raw_step)
        if len(step_notes) == 1:
            singles = _candidates_for_pitch(pretty_midi.note_name_to_number(step_notes[0]["pitch"]))
            if not singles:
                # Same policy as _make_voiceable's chord-note drops: an
                # unplayable note is skipped with a warning, not a failed job.
                logger.warning(
                    "map_fretboard: dropped %s (start %.2fs, amplitude %s): "
                    "outside the playable range (frets 0-%d, standard tuning)",
                    step_notes[0]["pitch"],
                    step_notes[0]["start_time"],
                    step_notes[0].get("amplitude", "n/a"),
                    MAX_FRET,
                )
                continue
            candidates = [[c] for c in singles]
        else:
            candidates = [voicing]
        steps.append(step_notes)
        step_candidates.append(candidates)

    if not step_candidates:
        return [], groups

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

    return result_notes, groups


@app.task(name="map_fretboard", soft_time_limit=120)
def map_fretboard(job_id: str) -> str:
    try:
        update_job(job_id, status="processing", stage="mapping")

        job = get_job(job_id)
        raw_note_events = job.get("raw_note_events")
        if raw_note_events is None:
            raise ValueError(f"job {job_id} has no raw_note_events; transcribe must run first")
        tempo_bpm = job.get("tempo_bpm")
        if tempo_bpm is None:
            raise ValueError(f"job {job_id} has no tempo_bpm; transcribe must run first")

        logger.info("map_fretboard: job %s mapping %d note(s)", job_id, len(raw_note_events))

        mapped_notes = map_notes_to_positions(raw_note_events)
        duration_seconds = max((note["end_time"] for note in mapped_notes), default=0.0)

        result = {
            "job_id": job_id,
            "duration_seconds": duration_seconds,
            # Estimated by transcribe (librosa beat tracking); see
            # _estimate_tempo_bpm there for reliability caveats.
            "tempo_bpm": tempo_bpm,
            "notes": mapped_notes,
        }

        # stage only describes in-progress work, so it's cleared once done. On
        # failure it's left as-is so the status shows where the job stopped.
        update_job(job_id, status="done", stage=None, result=result)
        logger.info("map_fretboard: job %s done with %d mapped note(s)", job_id, len(mapped_notes))
    except Exception as exc:
        logger.exception("map_fretboard failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"map_fretboard failed: {exc}")
        raise
    return job_id
