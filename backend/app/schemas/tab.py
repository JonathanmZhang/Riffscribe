from typing import List

from pydantic import BaseModel


class Note(BaseModel):
    string: int
    fret: int
    start_time: float
    end_time: float
    pitch: str


class ChordSegment(BaseModel):
    start: float
    end: float
    name: str


class TabResult(BaseModel):
    job_id: str
    duration_seconds: float
    tempo_bpm: float
    notes: List[Note]
    # Empty for jobs finished before chord names existed.
    chords: List[ChordSegment] = []
    # Beat and bar-start (downbeat) times in seconds. Empty for jobs
    # finished before beat tracking existed, or if it failed.
    beats: List[float] = []
    downbeats: List[float] = []
    # Start time (seconds) of each measure of the MusicXML export, for the
    # job's current tempo_factor / bar_offset_beats (tasks/rhythm
    # .measure_starts). Not stored: the backend derives it on every read.
    # The first can be slightly negative (a bar starting before the audio).
    bars: List[float] = []
