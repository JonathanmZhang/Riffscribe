from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, StrictInt

from app.schemas.tab import TabResult


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


# Which pipeline task is running (set by each worker task when it starts).
# Separate from status, which stays the fixed four-value enum. None when
# queued or done; left at the failing stage when a job fails.
JobStage = Literal["ingesting", "separating", "transcribing", "mapping"]

# Guitar separation, for isolate_guitar jobs. Quality is what was asked for:
# "standard" = Demucs (CPU), "high" = Mega 53 (NVIDIA GPU). Separator is
# what produced the stem; it differs from the quality after a fallback.
SeparationQuality = Literal["standard", "high"]
Separator = Literal["demucs", "mega53"]

# What a synth plays the MusicXML export with (tasks/musicxml.PLAYBACK_TONES):
# the part's General MIDI program. A request option, not stored on the job.
PlaybackTone = Literal["clean", "overdriven", "distorted", "acoustic"]

# Job-level notation overrides (tasks/rhythm.TEMPO_FACTORS / BAR_OFFSETS).
TempoFactor = Literal[0.5, 1.0, 2.0]
BarOffset = Literal[0, 1, 2, 3]
# Where on the neck the tab is placed (tasks/fretmap.neck_window): "auto"
# (the mapper's own choice), "open" (frets 0-4, open strings allowed), or a
# centre fret N for frets N-3..N+3 (no open strings), N in
# fretmap.NECK_CENTRES (1-17).
NeckPosition = Union[Literal["auto", "open"], Annotated[StrictInt, Field(ge=1, le=17)]]


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    error: Optional[str] = None
    result: Optional[TabResult] = None
    isolate_guitar: bool = False
    stage: Optional[JobStage] = None
    # True once the separated guitar stem exists; served by GET /jobs/{id}/stem.
    stem_available: bool = False
    # isolate_guitar jobs only. separator is set once the stem exists.
    # separation_note says why a "high" job was separated with Demucs.
    separation_quality: Optional[SeparationQuality] = None
    separator: Optional[Separator] = None
    separation_note: Optional[str] = None
    # Notation overrides (PATCH /jobs/{id}); they change the bars and the
    # MusicXML export, never the transcription.
    tempo_factor: TempoFactor = 1.0
    bar_offset_beats: BarOffset = 0
    # Changes result.notes' strings and frets (and the export's TAB staff),
    # never which notes there are.
    neck_position: NeckPosition = "auto"


class JobOverrides(BaseModel):
    """PATCH /jobs/{id} body; fields left out keep their current value.
    tempo_factor: 0.5 halves beat_this's beats (every other one), 2 doubles
    them (fast songs often come out at half tempo). bar_offset_beats: moves
    the bar lines later by this many beats. neck_position: re-maps the
    stored notes onto the fretboard inside that fret window."""

    tempo_factor: Optional[TempoFactor] = None
    bar_offset_beats: Optional[BarOffset] = None
    neck_position: Optional[NeckPosition] = None


class JobRerun(BaseModel):
    """POST /jobs/{id}/rerun body: transcribe the same source audio again as
    a new job, with these isolation settings."""

    isolate_guitar: bool = True
    separation_quality: SeparationQuality = "standard"


class JobRecord(BaseModel):
    """Internal representation persisted in Redis."""

    job_id: str
    status: JobStatus
    error: Optional[str] = None
    result: Optional[TabResult] = None
    isolate_guitar: bool = False
    # None unless isolate_guitar.
    separation_quality: Optional[SeparationQuality] = None
