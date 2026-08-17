"""Step 4-5: sound-event tagging, speech detection, and change signals.

Runs a pretrained sound-event classifier (see audio_classifier.py -- PANNs
Cnn14 today, swappable later) across the recording in overlapping windows,
producing a time-ordered series of WindowAnalysis records. Each one carries:

  - the sound-event labels detected in that window (for diversity/rarity)
  - whether speech was detected anywhere among those labels, not only as the
    top one (so Step 6 can avoid it even when it's a quieter background sound)
  - an embedding vector (so Step 6 can measure how much the soundscape
    changes from one moment to the next -- novelty and contrast)

This module only detects and describes what is acoustically present. It does
not judge importance, rank, or select candidates -- that happens in Step 6.

Window length (WINDOW_SECONDS) is 4.0s rather than a shorter slice on
purpose: PANNs Cnn14 (like most AudioSet taggers) was trained on ~10s clips,
and classification confidence measurably drops on much shorter windows,
especially for events with an attack/decay shape (a door closing, footsteps)
that need a bit of time to fully register.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .audio_classifier import get_classifier
from .segmentation import TARGET_SAMPLE_RATE, AudioChunk, decode_to_waveform, segment_recording

WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0
LABEL_SCORE_THRESHOLD = 0.10  # minimum confidence to count a label as "present"
TOP_K_LABELS = 5

# Verbal / language-bearing human vocalizations -- the goal is NO SPEECH, not NO
# HUMAN ACTIVITY, so this is deliberately narrower than AudioSet's full "Human
# sounds" branch. Included: anything that is, or is functionally equivalent to,
# spoken language a listener could make out words from -- ordinary speech and its
# speaker-identity variants, conversation, narration, whispering (quiet but still
# words), shouting/yelling (usually a word or name), and crowd/hubbub speech noise
# (many overlapping voices -- AudioSet's own label for class 70 is literally
# "speech noise, speech babble"). Deliberately EXCLUDED, and treated as ordinary
# candidate-eligible sound: non-verbal vocalizations that carry no language content
# -- laughter, crying, screaming, singing/chanting/humming, coughing, breathing,
# and similar "Human sounds" classes. A laugh or a scream is human, but it isn't
# someone talking, so it doesn't get filtered out on that basis alone.
SPEECH_LABELS = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Babbling", "Speech synthesizer",
    "Shout", "Yell", "Whispering", "Chatter", "Crowd",
    "Hubbub, speech noise, speech babble",
}

# Used both to flag an analysis window as speech-containing (see has_speech below)
# and to re-check a final exported clip (see clip_contains_speech below). Lower
# than a "this is clearly the dominant sound" bar on purpose: this only needs to
# notice speech is IDENTIFIABLE somewhere, not that it's the single loudest thing.
SPEECH_PRESENCE_THRESHOLD = 0.15


@dataclass
class WindowAnalysis:
    start_seconds: float
    labels: list  # [(label, score), ...] above threshold, sorted by score desc
    has_speech: bool  # any SPEECH_LABELS entry present in `labels`, not just the top one
    embedding: np.ndarray  # classifier embedding, for novelty/contrast in Step 6


def _windows_for_chunk(chunk: AudioChunk) -> list[tuple[float, np.ndarray]]:
    """Slice one chunk's samples into (local_start_seconds, samples) windows."""
    window_len = int(WINDOW_SECONDS * TARGET_SAMPLE_RATE)
    hop_len = int(HOP_SECONDS * TARGET_SAMPLE_RATE)
    total = len(chunk.samples)

    if total <= window_len:
        padded = np.pad(chunk.samples, (0, window_len - total))
        return [(0.0, padded)]

    windows = []
    start = 0
    while start < total:
        end = min(start + window_len, total)
        segment = chunk.samples[start:end]
        if len(segment) < window_len:
            segment = np.pad(segment, (0, window_len - len(segment)))
        windows.append((start / TARGET_SAMPLE_RATE, segment))
        if end == total:
            break
        start += hop_len
    return windows


def analyze_recording(
    filepath: str,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> list[WindowAnalysis]:
    """Run windowed sound-event analysis across the full recording.

    Returns WindowAnalysis records in chronological order, deduplicated
    across the small overlap between consecutive Step-3 chunks.
    """
    classifier = get_classifier()
    chunks = segment_recording(filepath)

    results: list[WindowAnalysis] = []
    last_kept_start = -float("inf")
    min_spacing = HOP_SECONDS * 0.5

    for chunk_index, chunk in enumerate(chunks):
        local_windows = _windows_for_chunk(chunk)
        batch = np.stack([w[1] for w in local_windows])
        clipwise_output, embeddings = classifier.classify_batch(batch)

        for (local_start, _samples), scores, embedding in zip(local_windows, clipwise_output, embeddings):
            absolute_start = chunk.start_seconds + local_start
            if absolute_start <= last_kept_start + min_spacing:
                continue
            last_kept_start = absolute_start

            top_idx = np.argsort(scores)[::-1][:TOP_K_LABELS]
            labels = [
                (classifier.labels[i], float(scores[i]))
                for i in top_idx
                if scores[i] >= LABEL_SCORE_THRESHOLD
            ]

            results.append(
                WindowAnalysis(
                    start_seconds=absolute_start,
                    labels=labels,
                    # Any detected label being speech is enough to flag the window --
                    # not just whether speech happens to be the single highest score.
                    # A quiet "person talking" under a louder environmental sound would
                    # still show up here as long as it cleared LABEL_SCORE_THRESHOLD.
                    has_speech=any(label in SPEECH_LABELS for label, _ in labels),
                    embedding=embedding,
                )
            )

        if progress_callback is not None:
            progress_callback((chunk_index + 1) / len(chunks))

    return results


# Sub-windowing for the final-clip speech recheck below. Deliberately shorter than
# the main WINDOW_SECONDS analysis window: classifying the whole ~5-6.5s clip at
# once lets a few seconds of louder environmental sound dilute a brief run of
# speech under the SPEECH_PRESENCE_THRESHOLD average. A short sub-window isolates
# just the speech instead of averaging it away.
SPEECH_CHECK_WINDOW_SECONDS = 2.0
SPEECH_CHECK_HOP_SECONDS = 1.0


def _fixed_sub_windows(waveform: np.ndarray, window_seconds: float, hop_seconds: float, sample_rate: int) -> list:
    """Slice a waveform into fixed-length, zero-padded, possibly-overlapping windows."""
    window_len = int(window_seconds * sample_rate)
    hop_len = int(hop_seconds * sample_rate)
    total = len(waveform)

    if total <= window_len:
        return [np.pad(waveform, (0, window_len - total))]

    segments = []
    start = 0
    while start < total:
        end = min(start + window_len, total)
        segment = waveform[start:end]
        if len(segment) < window_len:
            segment = np.pad(segment, (0, window_len - len(segment)))
        segments.append(segment)
        if end == total:
            break
        start += hop_len
    return segments


def clip_contains_speech(filepath: str) -> bool:
    """Check a final EXPORTED clip (not a recording-wide analysis window) for any
    identifiable speech, anywhere in it -- not just as the single dominant sound,
    and not just on average across the whole clip.

    Clip extraction (Step 7) repositions the ~5-6.5s boundary using pure audio
    energy, within a couple of seconds of whatever window was originally
    classified, so it can drift onto adjacent speech that analyze_recording()
    never saw -- this is the safety net that looks at the clip that will actually
    be shown/played. It re-classifies the clip in short SPEECH_CHECK_WINDOW_SECONDS
    sub-windows rather than as one whole-clip average, so a brief or quiet run of
    speech under a louder environmental sound is still caught: any sub-window
    scoring above SPEECH_PRESENCE_THRESHOLD on any speech class fails the clip.
    """
    classifier = get_classifier()
    waveform = decode_to_waveform(filepath, sample_rate=TARGET_SAMPLE_RATE)
    if len(waveform) == 0:
        return False

    speech_indices = [i for i, label in enumerate(classifier.labels) if label in SPEECH_LABELS]
    if not speech_indices:
        return False

    segments = _fixed_sub_windows(waveform, SPEECH_CHECK_WINDOW_SECONDS, SPEECH_CHECK_HOP_SECONDS, TARGET_SAMPLE_RATE)
    batch = np.stack(segments)
    scores, _embeddings = classifier.classify_batch(batch)
    return bool(np.max(scores[:, speech_indices]) >= SPEECH_PRESENCE_THRESHOLD)


def classify_clip(filepath: str) -> list:
    """Re-classify a FINAL EXPORTED CLIP as a whole, for display purposes.

    Clip extraction (Step 7) can reposition a candidate's boundaries by a couple
    of seconds from wherever it was originally classified (see
    clipping._clip_bounds), so the labels captured during analyze_recording()
    may no longer accurately describe what's actually inside the exported clip.
    This re-classifies the exact audio that will be shown/played, so displayed
    labels always correspond to the final clip rather than the original
    analysis window. Uses the same TOP_K_LABELS / LABEL_SCORE_THRESHOLD as
    analyze_recording, so "detected" means the same thing everywhere in the
    pipeline. Does not affect candidate ranking/selection -- that has already
    happened by the time this runs (see pipeline.generate_verified_candidates).
    """
    classifier = get_classifier()
    waveform = decode_to_waveform(filepath, sample_rate=TARGET_SAMPLE_RATE)
    if len(waveform) == 0:
        return []

    scores, _embedding = classifier.classify_one(waveform)
    top_idx = np.argsort(scores)[::-1][:TOP_K_LABELS]
    return [
        (classifier.labels[i], float(scores[i]))
        for i in top_idx
        if scores[i] >= LABEL_SCORE_THRESHOLD
    ]
