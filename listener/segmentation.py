"""Step 3: split long recordings into manageable chunks for analysis.

A 15-45 minute recording is too large to hand a sound-event model in one
pass. This module decodes the original recording once (at a fixed sample
rate) and slices it into overlapping chunks, each remembering its offset in
the original recording so later steps can map detections back to absolute
timestamps.
"""
import subprocess
from dataclasses import dataclass

import numpy as np

TARGET_SAMPLE_RATE = 32_000  # matches the PANNs Cnn14 sound-event model used in Step 4
CHUNK_SECONDS = 60.0
OVERLAP_SECONDS = 2.0  # avoids losing events that straddle a chunk boundary


@dataclass
class AudioChunk:
    samples: np.ndarray  # mono float32 at TARGET_SAMPLE_RATE
    start_seconds: float  # offset of this chunk within the original recording


def decode_to_waveform(filepath: str, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode any supported audio file to a mono float32 waveform via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", filepath,
            "-ac", "1", "-ar", str(sample_rate),
            "-f", "f32le", "-",
        ],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def decode_region_to_waveform(
    filepath: str, start_seconds: float, duration_seconds: float, sample_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Decode just a slice of a recording, for cheap local analysis (e.g. Step 7 clip positioning)."""
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-ss", str(max(0.0, start_seconds)), "-t", str(duration_seconds), "-i", filepath,
            "-ac", "1", "-ar", str(sample_rate),
            "-f", "f32le", "-",
        ],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def chunk_waveform(
    waveform: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> list[AudioChunk]:
    """Slice a waveform into overlapping chunks, each tagged with its offset."""
    chunk_len = int(chunk_seconds * sample_rate)
    hop_len = int((chunk_seconds - overlap_seconds) * sample_rate)
    total = len(waveform)

    if total <= chunk_len:
        return [AudioChunk(samples=waveform, start_seconds=0.0)]

    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_len, total)
        chunks.append(AudioChunk(samples=waveform[start:end], start_seconds=start / sample_rate))
        if end == total:
            break
        start += hop_len
    return chunks


def segment_recording(filepath: str) -> list[AudioChunk]:
    """Decode a recording and split it into overlapping analysis chunks."""
    waveform = decode_to_waveform(filepath)
    return chunk_waveform(waveform)
