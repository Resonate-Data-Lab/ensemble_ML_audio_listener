"""Step 7: extract ~5-second clips from the ORIGINAL recording.

Each candidate from Step 6 marks where a sound-event window was detected,
but that window's exact boundaries aren't necessarily where the sonic event
itself starts or ends. This module looks at the energy envelope of a small
region of the original audio around each candidate to find the event's real
edges, then extracts a clip that keeps the event whole while staying close
to the ~5 second target (spec section 9) -- rather than blindly cutting at
a fixed offset.

Clips are cut directly from the ORIGINAL uploaded file, not from the
resampled copies used for analysis, so the original stays the source of
truth (spec section 19).
"""
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .segmentation import decode_region_to_waveform
from .selection import Candidate

CLIP_DIR = Path(__file__).resolve().parent.parent / "data" / "clips"

TARGET_CLIP_SECONDS = 5.0
MAX_CLIP_SECONDS = 6.5  # spec: "keep clips short and relatively consistent"
SEARCH_PADDING_SECONDS = 2.5  # how far before/after the detected window to look for the event's real edges
ENVELOPE_SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.05
ACTIVE_THRESHOLD_RATIO = 1.6  # a frame counts as "active" once it's this many times the local noise floor
EDGE_PAD_SECONDS = 0.15  # small pad around a found event so cuts don't feel abrupt


@dataclass
class Clip:
    start_seconds: float
    end_seconds: float
    filepath: str


def _energy_envelope(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_len = max(1, int(FRAME_SECONDS * sample_rate))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return np.array([np.sqrt(np.mean(samples**2) + 1e-12)])
    trimmed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12)


def _find_event_span_seconds(
    envelope: np.ndarray, region_start: float, seed_start: float, seed_end: float
):
    """Find the contiguous active span (in absolute seconds) overlapping [seed_start, seed_end]."""
    floor = np.percentile(envelope, 20)
    threshold = floor * ACTIVE_THRESHOLD_RATIO
    active = envelope > threshold

    seed_start_frame = max(0, int((seed_start - region_start) / FRAME_SECONDS))
    seed_end_frame = min(len(active), int((seed_end - region_start) / FRAME_SECONDS) + 1)
    seed_active = [i for i in range(seed_start_frame, seed_end_frame) if active[i]]
    if not seed_active:
        return None

    start_frame = min(seed_active)
    while start_frame > 0 and active[start_frame - 1]:
        start_frame -= 1
    end_frame = max(seed_active)
    while end_frame + 1 < len(active) and active[end_frame + 1]:
        end_frame += 1

    return (
        region_start + start_frame * FRAME_SECONDS,
        region_start + (end_frame + 1) * FRAME_SECONDS,
    )


def _default_window(center: float, recording_duration: float, length: float = TARGET_CLIP_SECONDS) -> tuple:
    start = max(0.0, center - length / 2)
    end = min(recording_duration, start + length)
    start = max(0.0, end - length)  # re-clamp if we hit the recording's end
    return start, end


def _clip_bounds(
    recording_filepath: str, candidate: Candidate, window_seconds: float, recording_duration: float
) -> tuple:
    """Decide the [start, end] seconds to extract for one candidate."""
    window_center = candidate.start_seconds + window_seconds / 2
    search_start = max(0.0, candidate.start_seconds - SEARCH_PADDING_SECONDS)
    search_end = min(recording_duration, candidate.start_seconds + window_seconds + SEARCH_PADDING_SECONDS)

    if search_end - search_start < 0.5:
        return _default_window(window_center, recording_duration)

    try:
        region = decode_region_to_waveform(
            recording_filepath, search_start, search_end - search_start, sample_rate=ENVELOPE_SAMPLE_RATE
        )
        envelope = _energy_envelope(region, ENVELOPE_SAMPLE_RATE)
        span = _find_event_span_seconds(
            envelope, search_start, candidate.start_seconds, candidate.start_seconds + window_seconds
        )
    except subprocess.CalledProcessError:
        span = None

    if span is None:
        return _default_window(window_center, recording_duration)

    event_start = max(0.0, span[0] - EDGE_PAD_SECONDS)
    event_end = min(recording_duration, span[1] + EDGE_PAD_SECONDS)
    event_length = event_end - event_start

    if event_length <= TARGET_CLIP_SECONDS:
        # short event: give it a full ~5s of context, centered on where it happened
        return _default_window((event_start + event_end) / 2, recording_duration)

    if event_length > MAX_CLIP_SECONDS:
        # long/ongoing sound: keep it short and consistent rather than cutting the whole thing
        event_end = event_start + MAX_CLIP_SECONDS

    return event_start, min(recording_duration, event_end)


def _extract_clip(original_filepath: str, start: float, end: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(start), "-t", str(end - start), "-i", original_filepath,
            "-c:a", "pcm_s16le", str(output_path),
        ],
        check=True,
    )


def extract_clips(
    recording_filepath: str, recording_duration: float, candidates: list[Candidate], window_seconds: float
) -> list[Clip]:
    """Extract one ~5s clip per candidate from the original recording, in order."""
    run_id = uuid.uuid4().hex[:8]
    clips = []
    for i, candidate in enumerate(candidates, start=1):
        start, end = _clip_bounds(recording_filepath, candidate, window_seconds, recording_duration)
        output_path = CLIP_DIR / f"{run_id}_candidate_{i:02d}.wav"
        _extract_clip(recording_filepath, start, end, output_path)
        clips.append(Clip(start_seconds=start, end_seconds=end, filepath=str(output_path)))

    return clips
