"""Audio sound-event classification abstraction.

Wraps whichever pretrained classifier actually detects and labels sound
events behind one small interface, so the rest of Listener (windowing in
analysis.py, selection, description) never depends on a specific model.
Swapping to a different environmental-sound classifier later (YAMNet, AST,
BEATs, or anything else) means writing one new class here, not touching
how recordings get windowed, ranked, or described.

The concrete classifier used today is PANNs Cnn14 (trained on AudioSet,
527 classes) -- a real pretrained neural network, not a template or
fallback. It was already working before this abstraction existed; this
file only gives it a swappable seam, it doesn't change what it does.
"""
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class AudioClassifier(ABC):
    """A pretrained sound-event tagger: waveform in, (label scores, embedding) out."""

    sample_rate: int
    labels: list  # class vocabulary, in output-index order

    @abstractmethod
    def classify_batch(self, waveforms: np.ndarray) -> tuple:
        """Classify a batch of equal-length waveforms.

        waveforms: shape (batch, samples), mono float32 at self.sample_rate.
        Returns (scores, embeddings): scores shape (batch, n_classes) --
        one probability per label in self.labels -- and embeddings shape
        (batch, embedding_dim), a general-purpose representation of each
        clip's acoustic content (used for novelty/contrast, not labeling).
        """

    def classify_one(self, waveform: np.ndarray) -> tuple:
        """Classify a single waveform of any length. Returns (scores, embedding)."""
        scores, embeddings = self.classify_batch(waveform[None, :])
        return scores[0], embeddings[0]


class PannsCnn14Classifier(AudioClassifier):
    """PANNs Cnn14: a CNN trained on AudioSet's 527 sound-event classes."""

    sample_rate = 32_000

    # panns_inference downloads these itself via `wget` on first import, which
    # isn't installed on every machine (e.g. macOS by default). Fetching them
    # ourselves via urllib first makes that internal download a no-op.
    _PANNS_DATA_DIR = Path.home() / "panns_data"
    _LABELS_CSV = _PANNS_DATA_DIR / "class_labels_indices.csv"
    _CHECKPOINT_PATH = _PANNS_DATA_DIR / "Cnn14_mAP=0.431.pth"
    _LABELS_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
    _CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

    def __init__(self):
        self._ensure_assets()
        from panns_inference import AudioTagging, labels as audioset_labels

        self.labels = audioset_labels
        self._tagger = AudioTagging(checkpoint_path=str(self._CHECKPOINT_PATH), device="cpu")

    @classmethod
    def _ensure_assets(cls) -> None:
        cls._PANNS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not cls._LABELS_CSV.exists():
            urllib.request.urlretrieve(cls._LABELS_CSV_URL, cls._LABELS_CSV)
        if not cls._CHECKPOINT_PATH.exists() or cls._CHECKPOINT_PATH.stat().st_size < 3e8:
            urllib.request.urlretrieve(cls._CHECKPOINT_URL, cls._CHECKPOINT_PATH)

    def classify_batch(self, waveforms: np.ndarray) -> tuple:
        scores, embeddings = self._tagger.inference(waveforms)
        return scores, embeddings


_classifier = None


def get_classifier() -> AudioClassifier:
    """Return the shared classifier instance, constructing it (and downloading
    its assets/weights on first use) lazily. Swap the concrete class returned
    here to change classifiers without touching any caller."""
    global _classifier
    if _classifier is None:
        _classifier = PannsCnn14Classifier()
    return _classifier
