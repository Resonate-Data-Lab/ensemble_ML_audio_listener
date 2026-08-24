"""Step 4-5: audio representation extraction, speech gating, and change signals.

Version 6. Runs a pretrained sound classifier (see audio_classifier.py -- PANNs
Cnn14 today, swappable later) across the recording in overlapping windows, but
-- as of V6 -- ONLY for two things:

  1. its embedding (a general-purpose representation of each window's acoustic
     content, used downstream for novelty/contrast/diversity -- see selection.py)
  2. a narrow, binary check of whether any of its speech-related output classes
     crosses a presence threshold (see _has_speech below)

V5 and earlier also kept PANNs' full top-K AudioSet label list per window and
used label identity/confidence/frequency as scoring features (clarity, rarity,
layering). V6 removes that entirely: the goal is for the system to surface
distinctive sonic moments without needing to know or name what a sound is, so
"what does PANNs think this is called" no longer feeds selection anywhere.
Three signal-level features computed directly from each window's raw audio --
layering (spectral band entropy), distinctiveness (spectral crest), and
temporal interest (intra-window energy flux) -- replace what label counts and
label confidence used to stand in for. See selection.py's module docstring
for how these combine with the embedding-based novelty/contrast into a score,
and for how V6.2 groups near-identical adjacent windows into one sonic region
before ranking (a V6.1 attempt at a sixth "Sonic Interpretability" dimension
was tried and reverted here -- see selection.py's docstring for why).

The one place PANNs' classification output (not just its embedding) is still
consulted is the speech gate below. This is a deliberate, flagged exception,
not an oversight -- see _has_speech's docstring for why it stays and what it
is scoped to.

Window length (WINDOW_SECONDS) is 4.0s rather than a shorter slice on
purpose: PANNs Cnn14 (like most AudioSet taggers) was trained on ~10s clips,
and its embedding quality measurably drops on much shorter windows,
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
#
# FLAGGED (see module docstring / Step 6 design notes, V6): this set of class
# NAMES is still matched against PANNs' AudioSet class vocabulary, which makes
# the speech gate the one remaining place classification output (not just the
# embedding) is used. A fully label-independent speech detector (e.g. a
# dedicated VAD model) would remove this too, but that means adding a new
# model dependency, which V6 was explicitly asked not to do without a
# separate decision -- so this stays, narrowly scoped to "is any of these
# specific classes present", never exposed as a general label and never used
# for scoring. If you want this gone too, that is a real follow-up, not a
# free removal.
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

# ---------------------------------------------------------------------------
# Signal-level feature parameters (V6). All computed directly from a window's
# raw samples -- no classifier, no labels -- and are only min-max-normalized
# into final 0-1 scores later, across the whole eligible pool (see
# selection.build_candidate_pool). Raw values stored on WindowAnalysis are
# therefore recording-relative, not universal/absolute measures.
# ---------------------------------------------------------------------------

# Layering: how many independent frequency bands carry real, comparable energy
# at once. Roughly sub-bass through upper-mid/presence range for
# environmental recordings; deliberately coarse (6 bands) rather than a full
# spectrogram, since the question is "how spread out is the energy", not
# precise spectral shape.
LAYERING_BANDS_HZ = [(20, 150), (150, 400), (400, 1000), (1000, 2500), (2500, 6000), (6000, 16000)]

# Temporal interest: split each window into short sub-frames and look at how
# much the energy envelope moves within the window itself (as opposed to
# novelty/contrast, which compare BETWEEN windows). 0.5s sub-frames give 8
# per 4s window -- enough to catch an onset/offset or a rhythmic repeat
# without being so short that it's just measuring noise-floor jitter.
TEMPORAL_SUBFRAME_SECONDS = 0.5


@dataclass
class WindowAnalysis:
    start_seconds: float
    has_speech: bool  # narrow speech-class check only -- see SPEECH_LABELS docstring above
    embedding: np.ndarray  # classifier embedding, for novelty/contrast/diversity/regions in Step 6
    # Raw (un-normalized) signal-level features -- see the parameter block
    # above. Normalized into final 0-1 scores in selection.build_candidate_pool.
    layering_raw: float
    distinctiveness_raw: float
    temporal_interest_raw: float


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


def _speech_class_indices(classifier) -> np.ndarray:
    return np.array([i for i, label in enumerate(classifier.labels) if label in SPEECH_LABELS])


def _has_speech(scores: np.ndarray, speech_indices: np.ndarray) -> bool:
    """Narrow binary check: does any speech-class score cross the presence bar?

    Deliberately does NOT extract or expose a general label list -- this reads
    only the fixed subset of class scores named in SPEECH_LABELS, the same way
    clip_contains_speech() below checks a final exported clip. See the
    SPEECH_LABELS docstring for why this one classification-based check
    remains while general label-based scoring does not.
    """
    if speech_indices.size == 0:
        return False
    return bool(np.max(scores[speech_indices]) >= SPEECH_PRESENCE_THRESHOLD)


def _magnitude_spectrum(samples: np.ndarray) -> np.ndarray:
    """Shared FFT magnitude spectrum, computed once per window and reused by
    every spectral feature below (layering, distinctiveness) instead of each
    recomputing its own."""
    return np.abs(np.fft.rfft(samples * np.hanning(len(samples))))


def _band_energy_entropy(spectrum: np.ndarray, freqs: np.ndarray) -> float:
    """Layering (raw): Shannon entropy of energy spread across LAYERING_BANDS_HZ.

    High when energy is spread roughly evenly across many bands (several
    concurrent acoustic components/textures at once -- e.g. broadband rain
    plus a tonal bird call plus low rumble). Low when one band dominates (a
    single, spectrally narrow source). Self-normalizing to [0, 1] via
    max-entropy division, so no recording-wide calibration is needed here --
    final pool-relative normalization still happens in build_candidate_pool
    for consistency with the other dimensions.
    """
    band_energy = []
    for lo, hi in LAYERING_BANDS_HZ:
        mask = (freqs >= lo) & (freqs < hi)
        band_energy.append(float(np.sqrt(np.mean(spectrum[mask] ** 2))) if mask.any() else 0.0)
    band_energy = np.asarray(band_energy)

    total = band_energy.sum()
    if total <= 1e-9:
        return 0.0
    p = band_energy[band_energy > 0] / total
    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = np.log(len(LAYERING_BANDS_HZ))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _spectral_crest(spectrum: np.ndarray) -> float:
    """Distinctiveness (raw): peak-to-mean ratio of the magnitude spectrum.

    High when a segment has a strong, well-defined spectral character (one or
    two prominent components clearly above the rest) -- a segment that
    "stands out" acoustically regardless of whether anything can name it. Low
    for a diffuse, undifferentiated texture where energy is smeared evenly
    across frequency. This is about acoustic prominence, never about
    classifier confidence.
    """
    mean = float(spectrum.mean()) + 1e-9
    return float(spectrum.max()) / mean


def _temporal_flux(samples: np.ndarray, sample_rate: int) -> float:
    """Temporal interest (raw): how much the energy envelope moves WITHIN
    this window, not between windows (that's novelty/contrast's job).

    Splits the window into ~8 sub-frames and takes the mean absolute
    frame-to-frame change in RMS -- high for something appearing, vanishing,
    or repeating during the window; near zero for static, unchanging texture
    (room tone, steady rain) even if that texture is loud.
    """
    frame_len = max(1, int(TEMPORAL_SUBFRAME_SECONDS * sample_rate))
    n_frames = len(samples) // frame_len
    if n_frames < 2:
        return 0.0
    trimmed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    envelope = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
    return float(np.mean(np.abs(np.diff(envelope))))


def analyze_recording(
    filepath: str,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> list[WindowAnalysis]:
    """Run windowed audio-representation analysis across the full recording.

    Returns WindowAnalysis records in chronological order, deduplicated
    across the small overlap between consecutive Step-3 chunks.
    """
    classifier = get_classifier()
    speech_indices = _speech_class_indices(classifier)
    chunks = segment_recording(filepath)

    results: list[WindowAnalysis] = []
    last_kept_start = -float("inf")
    min_spacing = HOP_SECONDS * 0.5

    for chunk_index, chunk in enumerate(chunks):
        local_windows = _windows_for_chunk(chunk)
        batch = np.stack([w[1] for w in local_windows])
        clipwise_output, embeddings = classifier.classify_batch(batch)

        for (local_start, samples), scores, embedding in zip(local_windows, clipwise_output, embeddings):
            absolute_start = chunk.start_seconds + local_start
            if absolute_start <= last_kept_start + min_spacing:
                continue
            last_kept_start = absolute_start

            spectrum = _magnitude_spectrum(samples)
            freqs = np.fft.rfftfreq(len(samples), 1.0 / TARGET_SAMPLE_RATE)

            results.append(
                WindowAnalysis(
                    start_seconds=absolute_start,
                    has_speech=_has_speech(scores, speech_indices),
                    embedding=embedding,
                    layering_raw=_band_energy_entropy(spectrum, freqs),
                    distinctiveness_raw=_spectral_crest(spectrum),
                    temporal_interest_raw=_temporal_flux(samples, TARGET_SAMPLE_RATE),
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

    Same flagged exception as _has_speech above: this is classification-based,
    kept narrowly for the speech gate only.
    """
    classifier = get_classifier()
    waveform = decode_to_waveform(filepath, sample_rate=TARGET_SAMPLE_RATE)
    if len(waveform) == 0:
        return False

    speech_indices = _speech_class_indices(classifier)
    if speech_indices.size == 0:
        return False

    segments = _fixed_sub_windows(waveform, SPEECH_CHECK_WINDOW_SECONDS, SPEECH_CHECK_HOP_SECONDS, TARGET_SAMPLE_RATE)
    batch = np.stack(segments)
    scores, _embeddings = classifier.classify_batch(batch)
    return bool(np.max(scores[:, speech_indices]) >= SPEECH_PRESENCE_THRESHOLD)
