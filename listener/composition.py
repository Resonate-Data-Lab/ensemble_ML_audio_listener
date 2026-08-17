"""Version 3: assemble the user's own trim/reorder/crossfade decisions into one file.

This module does not decide anything -- it just renders whatever sequence,
trims, and crossfade choices the person made in the editing workspace into
a single audio file. No AI sequencing, no automatic blending: a crossfade
only happens where the user explicitly turned one on for that transition.
"""
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .segmentation import decode_to_waveform

COMPOSITION_DIR = Path(__file__).resolve().parent.parent / "data" / "compositions"

CROSSFADE_SECONDS = 0.4  # subtle, per spec -- not adjustable by design

# Version 4: a hard concat (crossfade off) can splice clip A's end directly
# against clip B's start with no shaping, which produces an audible click/pop
# if neither edge happens to sit near zero amplitude. DECLICK_SECONDS is a very
# short linear fade at just that boundary -- NOT a crossfade: there is no time
# overlap between clips, both still play their full duration back-to-back, only
# the outermost ~10ms of each edge is ramped toward zero. This only touches the
# no-crossfade path; _crossfade() below is unchanged.
DECLICK_SECONDS = 0.01

# Version 4: onset/boundary awareness. Same energy-envelope method as
# clipping._find_event_span_seconds (same frame size, same floor*ratio
# threshold), applied here to an already-exported/trimmed clip to find where
# its OWN meaningful audio starts and where trailing near-silence begins.
# Only used to decide WHERE two clips should meet -- MAX_BOUNDARY_SILENCE_SECONDS
# is a deliberately generous bar so short natural pauses are never touched,
# only silence long enough to clearly be unused boundary material. This never
# changes how a transition blends: _crossfade() and _concat() below are called
# exactly as before, just possibly on a boundary-trimmed version of a clip.
ONSET_SAMPLE_RATE = 16_000
ONSET_FRAME_SECONDS = 0.05
ONSET_ACTIVE_THRESHOLD_RATIO = 1.6
MAX_BOUNDARY_SILENCE_SECONDS = 0.5
BOUNDARY_PAD_SECONDS = 0.15  # kept when trimming, so the cut itself isn't abrupt


def _usable_bounds(filepath: str) -> tuple:
    """Return (onset_seconds, trailing_silence_seconds, duration_seconds) for a
    clip: where its meaningful audio starts, how much near-silent material
    trails after the last active frame, and its total duration. If nothing in
    the clip clearly rises above the noise floor, returns (0.0, 0.0, duration)
    -- nothing to trim, leave it exactly as-is rather than guessing."""
    waveform = decode_to_waveform(filepath, sample_rate=ONSET_SAMPLE_RATE)
    duration = len(waveform) / ONSET_SAMPLE_RATE
    if len(waveform) == 0:
        return 0.0, 0.0, 0.0

    frame_len = max(1, int(ONSET_FRAME_SECONDS * ONSET_SAMPLE_RATE))
    n_frames = len(waveform) // frame_len
    if n_frames == 0:
        return 0.0, 0.0, duration
    trimmed = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
    envelope = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12)

    floor = np.percentile(envelope, 20)
    threshold = floor * ONSET_ACTIVE_THRESHOLD_RATIO
    active_indices = np.nonzero(envelope > threshold)[0]
    if len(active_indices) == 0:
        return 0.0, 0.0, duration

    onset = float(active_indices[0]) * ONSET_FRAME_SECONDS
    last_active_end = (float(active_indices[-1]) + 1) * ONSET_FRAME_SECONDS
    trailing_silence = max(0.0, duration - last_active_end)
    return onset, trailing_silence, duration


@dataclass
class CompositionItem:
    candidate_id: int
    clip_filepath: str
    clip_start_seconds: float  # this clip's position in the ORIGINAL recording (for logging)
    clip_end_seconds: float
    trim_start: float = 0.0  # offset within the clip itself, not the original recording
    trim_end: float = 0.0  # defaults to the clip's own duration; set by caller on init
    crossfade_into_next: bool = False


def _trim_clip(filepath: str, start: float, end: float, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(start), "-t", str(max(0.05, end - start)), "-i", filepath,
            "-c:a", "pcm_s16le", str(output_path),
        ],
        check=True,
    )


def _concat(a_path: Path, b_path: Path, output_path: Path, a_duration: float, b_duration: float) -> None:
    """Splice two clips with no time overlap. Applies a short declick fade at the
    join (see DECLICK_SECONDS) when both clips are long enough for it to make
    sense; otherwise falls back to a plain concat, unchanged from before."""
    declick = min(DECLICK_SECONDS, a_duration * 0.2, b_duration * 0.2)
    if declick > 0.002:
        fade_out_start = max(0.0, a_duration - declick)
        filter_complex = (
            f"[0:a]afade=t=out:st={fade_out_start}:d={declick}[a_faded];"
            f"[1:a]afade=t=in:st=0:d={declick}[b_faded];"
            f"[a_faded][b_faded]concat=n=2:v=0:a=1[out]"
        )
    else:
        filter_complex = "[0:a][1:a]concat=n=2:v=0:a=1[out]"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(a_path), "-i", str(b_path),
            "-filter_complex", filter_complex,
            "-map", "[out]", "-c:a", "pcm_s16le", str(output_path),
        ],
        check=True,
    )


def _crossfade(a_path: Path, b_path: Path, output_path: Path, fade_seconds: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(a_path), "-i", str(b_path),
            "-filter_complex", f"[0:a][1:a]acrossfade=d={fade_seconds}:c1=tri:c2=tri[out]",
            "-map", "[out]", "-c:a", "pcm_s16le", str(output_path),
        ],
        check=True,
    )


def build_composition(items: list, run_id: str) -> str:
    """Trim each item, then concatenate (or crossfade, per item.crossfade_into_next)
    them in the given order. Returns the path to the assembled file."""
    if not items:
        raise ValueError("composition has no items")

    COMPOSITION_DIR.mkdir(parents=True, exist_ok=True)
    work_id = uuid.uuid4().hex[:8]

    trimmed_paths = []
    for idx, item in enumerate(items):
        trimmed_path = COMPOSITION_DIR / f"{run_id}_{work_id}_trim_{idx}.wav"
        _trim_clip(item.clip_filepath, item.trim_start, item.trim_end, trimmed_path)
        trimmed_paths.append(trimmed_path)

    running_path = trimmed_paths[0]
    running_duration = items[0].trim_end - items[0].trim_start

    for idx in range(1, len(items)):
        next_path = trimmed_paths[idx]
        next_duration = items[idx].trim_end - items[idx].trim_start
        step_output = COMPOSITION_DIR / f"{run_id}_{work_id}_step_{idx}.wav"

        # V4: decide WHERE this pair should meet, based on each side's actual
        # usable audio -- only acts on CLEARLY excessive boundary silence (see
        # MAX_BOUNDARY_SILENCE_SECONDS), never on a short natural pause. This
        # only adjusts which samples of running_path/next_path get handed to
        # _crossfade()/_concat() below; it does not change how either of those
        # functions blends them -- that logic is untouched.
        _, running_trailing_silence, _ = _usable_bounds(str(running_path))
        if running_trailing_silence > MAX_BOUNDARY_SILENCE_SECONDS:
            adjusted_duration = max(0.05, running_duration - running_trailing_silence + BOUNDARY_PAD_SECONDS)
            adjusted_path = COMPOSITION_DIR / f"{run_id}_{work_id}_boundary_{idx}a.wav"
            _trim_clip(str(running_path), 0.0, adjusted_duration, adjusted_path)
            running_path = adjusted_path
            running_duration = adjusted_duration

        next_onset, _, _ = _usable_bounds(str(next_path))
        if next_onset > MAX_BOUNDARY_SILENCE_SECONDS:
            skip = next_onset - BOUNDARY_PAD_SECONDS
            adjusted_duration = next_duration - skip
            adjusted_path = COMPOSITION_DIR / f"{run_id}_{work_id}_boundary_{idx}b.wav"
            _trim_clip(str(next_path), skip, next_duration, adjusted_path)
            next_path = adjusted_path
            next_duration = adjusted_duration

        if items[idx - 1].crossfade_into_next:
            fade = min(CROSSFADE_SECONDS, running_duration * 0.9, next_duration * 0.9)
        else:
            fade = 0.0

        if fade > 0.05:
            _crossfade(running_path, next_path, step_output, fade)
            running_duration = running_duration + next_duration - fade
        else:
            _concat(running_path, next_path, step_output, running_duration, next_duration)
            running_duration = running_duration + next_duration

        running_path = step_output

    final_path = COMPOSITION_DIR / f"{run_id}_composition.wav"
    shutil.copyfile(running_path, final_path)
    return str(final_path)
