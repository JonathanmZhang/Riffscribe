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
