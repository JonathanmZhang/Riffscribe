import dataclasses
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


@dataclasses.dataclass(frozen=True)
class HandCosts:
    """Cost terms of the hand-position mapper (spec 3.5). The fretting hand
    sits with its index finger at some fret h and covers `span` frets
    (h .. h+span-1) without moving; all values are in the same arbitrary
    units. Tuned on all 12 EGSet12 performances with scripts/tune_hand_mapper.py
    (2-fold cross-validation by performance, same search: held-out position
    agreement 53.2/48.3/45.0% -> 55.8/49.7/44.7% clean/moderate/heavy vs the
    previous last-position mapper)."""

    span: int = 5
    # A fretted note one fret outside the span (pinky stretch, or the index
    # finger reaching back one fret).
    stretch_cost: float = 0.5
    # Any fretted note further outside the span, per note.
    out_of_span_cost: float = 10.0
    # Moving the hand: a fixed cost for any shift plus a per-fret cost, so the
    # tab stays in one region until the music requires a move.
    shift_cost: float = 8.0
    shift_per_fret: float = 0.5
    # Per open string played (open strings never constrain the hand).
    open_string_cost: float = 0.5
    # Open position: with the index finger at or below this fret, each open
    # string gets this bonus over playing the same pitch fretted, so open
    # notes ring the way guitarists play open chords. Only applies near the
    # nut, so it doesn't pull up-the-neck passages down.
    open_position_max_hand: int = 3
    open_position_bonus: float = 0.5
    # Weak preference for lower hand positions, per fret of h. Replaces the
    # old lexicographic lowest-fret tie-break, which pulled whole passages
    # toward the nut.
    fret_height_cost: float = 0.02
    # Chord voicings are only considered if their fretted notes span at most
    # this many frets (the exhaustive lowest-fret voicing is the fallback).
    max_voicing_span: int = 4


HAND_COSTS = HandCosts()
# Index-finger positions the DP considers.
HAND_POSITIONS = range(1, MAX_FRET - HAND_COSTS.span + 2)


def _chord_voicings(pitches: list[int], max_span: int) -> list[list[tuple[int, int]]]:
    """Every assignment of the pitches to distinct strings whose fretted
    notes span at most max_span frets (open strings excluded from the span),
    in the same order as `pitches`."""
    options = [_candidates_for_pitch(p) for p in pitches]
    order = sorted(range(len(pitches)), key=lambda i: len(options[i]))
    found: list[list[tuple[int, int]]] = []
    chosen: dict[int, tuple[int, int]] = {}

    def recurse(k: int, used: set[int], lo: int, hi: int) -> None:
        if k == len(order):
            found.append([chosen[i] for i in range(len(pitches))])
            return
        i = order[k]
        for string, fret in options[i]:
            if string in used:
                continue
            new_lo, new_hi = (min(lo, fret), max(hi, fret)) if fret > 0 else (lo, hi)
            if fret > 0 and new_hi - new_lo > max_span:
                continue
            chosen[i] = (string, fret)
            used.add(string)
            recurse(k + 1, used, new_lo, new_hi)
            used.discard(string)

    recurse(0, set(), MAX_FRET + 1, -1)
    return found


def _placement_cost(positions: list[tuple[int, int]], hand: int, costs: HandCosts) -> float:
    """Cost of playing one step's positions with the index finger at `hand`."""
    cost = costs.fret_height_cost * hand
    for _string, fret in positions:
        if fret == 0:
            cost += costs.open_string_cost
            if hand <= costs.open_position_max_hand:
                cost -= costs.open_position_bonus
        elif hand <= fret < hand + costs.span:
            continue
        elif fret == hand + costs.span or fret == hand - 1:
            cost += costs.stretch_cost
        else:
            cost += costs.out_of_span_cost
    return cost


def _shift_cost(previous_hand: int, hand: int, costs: HandCosts) -> float:
    if previous_hand == hand:
        return 0.0
    return costs.shift_cost + costs.shift_per_fret * abs(hand - previous_hand)


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


def map_notes_to_positions(notes: list[dict], costs: HandCosts = HAND_COSTS) -> list[dict]:
    """Maps note events onto guitar string/fret positions by following the
    player's fretting hand, per spec 3.5: a Viterbi-style dynamic program
    over (voicing, index-finger position) states. Playing notes outside the
    hand's span and shifting the hand cost (HandCosts), so the tab stays in
    one region of the neck until the music requires a move, the way
    guitarists play; open strings are free anywhere. Backpointers
    reconstruct the globally cheapest path, not a greedy choice.
    """
    return map_notes_with_steps(notes, costs)[0]


def map_notes_with_steps(notes: list[dict], costs: HandCosts = HAND_COSTS) -> tuple[list[dict], list[list[dict]]]:
    """map_notes_to_positions, also returning the groups (tab columns) the
    notes were placed in, including notes later dropped as unplayable. For
    the measurement scripts, so they see exactly the grouping the mapper
    used; the pipeline calls map_notes_to_positions.
    """
    if not notes:
        return [], []

    # step_candidates[i] lists the candidate position sets for step i: every
    # valid (string, fret) for a single note, and for a chord every voicing
    # on distinct strings within a playable span (falling back to the
    # exhaustive lowest-fret voicing when none fits). Chords that can't be
    # voiced at all are reduced first (notes dropped), so `steps` holds only
    # the notes that actually get placed.
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
            pitches = [pretty_midi.note_name_to_number(n["pitch"]) for n in step_notes]
            candidates = _chord_voicings(pitches, costs.max_voicing_span) or [voicing]
        steps.append(step_notes)
        step_candidates.append(candidates)

    if not step_candidates:
        return [], groups

    # dp[c][h]: cheapest cost of reaching candidate c at this step with the
    # index finger at hand position h. The shift cost depends only on the two
    # hand positions, so the best predecessor for each h is found once per
    # step over hand positions, not per candidate pair.
    hands = list(HAND_POSITIONS)
    dp = [[_placement_cost(c, h, costs) for h in hands] for c in step_candidates[0]]
    # Per step (from the second): for each hand index, the predecessor's hand
    # index, and for each hand index the best previous candidate there.
    back_hand: list[list[int]] = []
    back_candidate: list[list[int]] = []

    for i in range(1, len(step_candidates)):
        best_at_hand = [min(range(len(dp)), key=lambda c, hi=hi: dp[c][hi]) for hi in range(len(hands))]
        best_value_at_hand = [dp[best_at_hand[hi]][hi] for hi in range(len(hands))]
        arrive, arrive_from = [], []
        for hi, h in enumerate(hands):
            from_hi = min(range(len(hands)),
                          key=lambda pj: best_value_at_hand[pj] + _shift_cost(hands[pj], h, costs))
            arrive.append(best_value_at_hand[from_hi] + _shift_cost(hands[from_hi], h, costs))
            arrive_from.append(from_hi)
        back_hand.append(arrive_from)
        back_candidate.append(best_at_hand)
        dp = [[arrive[hi] + _placement_cost(c, h, costs) for hi, h in enumerate(hands)] for c in step_candidates[i]]

    # Backtrack from the cheapest final state.
    last_c, last_h = min(((c, hi) for c in range(len(dp)) for hi in range(len(hands))),
                         key=lambda state: dp[state[0]][state[1]])
    chosen_indices: list[int] = [0] * len(step_candidates)
    chosen_indices[-1] = last_c
    hand_index = last_h
    for i in range(len(step_candidates) - 1, 0, -1):
        hand_index = back_hand[i - 1][hand_index]
        chosen_indices[i - 1] = back_candidate[i - 1][hand_index]

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
