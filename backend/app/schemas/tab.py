from typing import List

from pydantic import BaseModel


class Note(BaseModel):
    string: int
    fret: int
    start_time: float
    end_time: float
    pitch: str


class TabResult(BaseModel):
    job_id: str
    duration_seconds: float
    tempo_bpm: float
    notes: List[Note]
