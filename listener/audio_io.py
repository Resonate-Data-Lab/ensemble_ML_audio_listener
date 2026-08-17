"""Step 1-2: accept an uploaded recording and read its basic properties.

The original file is saved as-is and treated as the source of truth for the
rest of the pipeline -- later steps extract clips from this file rather than
regenerating audio.
"""
import subprocess
import uuid
from pathlib import Path

from .models import Recording

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a"}


def save_uploaded_file(uploaded_file) -> Path:
    """Persist a Streamlit UploadedFile to disk and return its path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower()
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(uploaded_file.getvalue())
    return dest


def get_duration_seconds(filepath: Path) -> float:
    """Read duration via ffprobe, without fully decoding the file."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(filepath),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_recording(uploaded_file) -> Recording:
    path = save_uploaded_file(uploaded_file)
    duration = get_duration_seconds(path)
    return Recording(filename=uploaded_file.name, filepath=str(path), duration_seconds=duration)
