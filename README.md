# Listener

Surfaces candidate sound moments from an everyday audio recording and lets a person decide which ones are worth keeping.

Listener analyzes a long ambient recording, proposes short clips it judges to be acoustically distinct, and hands them to the listener to keep or discard. The person then trims, reorders, and crossfades what they kept into a short composition. The system does not decide what a sound means — it only surfaces possibilities.

## Installation

Requires Python 3.9 or later.

```bash
git clone https://github.com/Resonate-Data-Lab/ensemble_ML_audio_listener.git
cd ensemble_ML_audio_listener

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

PANNs CNN14 model weights download automatically on first run, so the first analysis takes longer than later ones.

## Usage

```bash
streamlit run app.py
```

Then in the browser:

1. Upload one recording (MP3, WAV, or M4A).
2. Press **Analyze Recording**. The system returns up to ten candidates of about five seconds each.
3. Play each one and choose **Keep** or **Discard**.
4. Optionally open **Explore an emotional interpretation** to hear a clip transformed through an affective direction, with the original kept alongside.
5. Open **Edit Selected Sounds** to trim, reorder, remove, restore, and crossfade.
6. Press **Preview Composition** to render the result.

Clips and compositions are written to `data/`.

## Features

- **Candidate surfacing** — proposes moments that stand out acoustically, so the person doesn't scrub through the whole recording.
- **Human curation** — nothing enters a composition without being kept by the person.
- **Editing workspace** — trim, reorder, remove, restore, crossfade. Removing a clip from the composition doesn't remove it from the kept set.
- **Affective transformation** — twelve optional directions (Calm, Nostalgic, Joyful, Melancholic, Tense, Mysterious, Dreamy, Lonely, Intimate, Energetic, Uneasy, Serene), applied per clip.
- **Research logging** — what the system surfaced and what the person kept are recorded separately.

Candidates are ranked on acoustic properties rather than classification labels: novelty, contrast, layering, distinctiveness, and temporal interest. These are engineering proxies and have not been validated against what people actually find worth keeping.

## Configuration

**Anthropic API key (optional)** — enables auto-generated textual descriptions of candidates. Everything else works without it.

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Project structure

```
ensemble_ML_audio_listener/
├── app.py                   # Streamlit web application interface
├── requirements.txt         # Python dependency specifications
├── README.md                # Project documentation
├── listener/                # Core Python library
│   ├── analysis.py          # Signal analysis, windowing, and scoring logic
│   ├── affect.py            # Affective tone transformation functions
│   ├── audio_classifier.py  # PANNs CNN model loading & embedding extraction
│   ├── audio_io.py          # Audio file loading, saving, and timestamp formatting
│   ├── clipping.py          # Audio clip extraction and trimming utilities
│   ├── composition.py       # Multi-clip composition rendering & crossfading
│   ├── pipeline.py          # Verification and candidate generation pipeline
│   ├── research_log.py      # Logging decision history and session snapshots
│   └── selection.py         # Candidate selection and re-ranking algorithms
└── data/                    # Temporary directory for saved clips & compositions
```

## Known limitations

Built for a single session and one recording of roughly thirty to forty-five minutes. It has no multi-user handling or session persistence. Acoustic boundaries are imperfect in continuous environments such as traffic or wind, repetitive recordings may not yield ten distinct candidates, affective transformations use fixed recipes, and audio is processed in mono.
