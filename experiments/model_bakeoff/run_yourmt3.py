"""YourMT3+ (YPTF.MoE+Multi, noPS - the Space's default checkpoint) on one
bake-off clip, on CPU at fp32. Runs in the bakeoff-yourmt3 image from
/opt/yourmt3: python /bakeoff/run_yourmt3.py <segment id>

Mirrors model_helper.transcribe() from the Space, but keeps the notes
instead of writing MIDI. It's multi-instrument: notes are kept only when
their program is a GM guitar (24-31); the full output's recall is also
recorded to show whether guitar was heard but labelled as another program.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.append("/opt/yourmt3/amt/src")
sys.path.append("/opt/yourmt3")

import time  # noqa: E402

import torch  # noqa: E402
import torchaudio  # noqa: E402

from common import clip_path, write_notes  # noqa: E402

ARGS = ["mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt", "-p", "2024",
        "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1",
        "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1",
        "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "32"]
GUITAR_PROGRAMS = range(24, 32)


def transcribe_notes(model, path: str) -> list:
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes

    audio, sr = torchaudio.load(path)
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg["sample_rate"])
    frames = model.audio_cfg["input_frames"]
    segments = torch.from_numpy(slice_padded_array(audio, frames, frames).astype("float32")).unsqueeze(1)
    with torch.no_grad():
        tokens, _ = model.inference_file(bsz=8, audio_segments=segments)
    starts = [frames * i / model.audio_cfg["sample_rate"] for i in range(segments.shape[0])]
    per_channel = []
    for ch in range(model.task_manager.num_decoding_channels):
        zipped, _, _ = model.task_manager.detokenize_list_batches([a[:, ch, :] for a in tokens], starts,
                                                                  return_events=True)
        notes, _ = merge_zipped_note_events_and_ties_to_notes(zipped)
        per_channel.append(notes)
    return mix_notes(per_channel)


def main(seg: str) -> None:
    from model_helper import load_model_checkpoint

    if os.environ.get("THREADS"):  # default: torch's own (all cores), as Basic Pitch/TF got
        torch.set_num_threads(int(os.environ["THREADS"]))
    t0 = time.perf_counter()
    model = load_model_checkpoint(args=ARGS, device="cpu")
    load = time.perf_counter() - t0

    path = clip_path(seg)
    transcribe_notes(model, path)  # warm-up, as for every model
    t0 = time.perf_counter()
    notes = transcribe_notes(model, path)
    infer = time.perf_counter() - t0
    duration = torchaudio.info(path).num_frames / torchaudio.info(path).sample_rate

    guitar = [n for n in notes if not n.is_drum and n.program in GUITAR_PROGRAMS]
    as_dict = lambda ns: [{"onset": float(n.onset), "end": float(n.offset), "midi": int(n.pitch)} for n in ns]  # noqa: E731
    programs = {}
    for n in notes:
        key = "drum" if n.is_drum else str(n.program)
        programs[key] = programs.get(key, 0) + 1
    write_notes("yourmt3-guitar", seg, duration, load, infer, as_dict(guitar), programs=programs,
                threads=torch.get_num_threads())
    write_notes("yourmt3-allinst", seg, duration, load, infer,
                as_dict([n for n in notes if not n.is_drum]), programs=programs)
    print(f"{seg}: {len(notes)} notes ({len(guitar)} guitar), programs {programs}, "
          f"{infer:.1f}s for {duration:.1f}s audio, load {load:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1])
