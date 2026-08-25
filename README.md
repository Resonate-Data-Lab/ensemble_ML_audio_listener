# LISTENER 🎧

**LISTENER** is an interactive AI-assisted audio exploration and composition tool built with Python and Streamlit.

It takes everyday environmental audio recordings and surfaces **candidate sonic moments** ("AI surfaces") based on signal-level acoustic properties (novelty, contrast, temporal interest, distinctiveness, and layering). Users can then review, interpret, apply affective sonic transformations (e.g., *Calm*, *Nostalgic*, *Dreamy*), edit, reorder, crossfade, and build custom audio compositions.

---

## 🌟 Key Features & Philosophy

- **Acoustic-First Candidate Scoring:** Candidate moments are detected and surfaced using PANNs (Cnn14 embeddings) for acoustic feature contrast and novelty—not by subjective or automated semantic labeling.
- **Human-Centric Interpretation:** AI surfaces the moments; humans decide what they mean, which ones to keep, and how to sequence or edit them.
- **Affective Tone Transformations:** Apply user-guided acoustic tone directions (*Calm*, *Nostalgic*, *Dreamy*, *Mysterious*, etc.) to transform audio clips without altering their underlying core identity.
- **Interactive Web Interface:** Streamlit UI allowing real-time playback, candidate selection, trimming, crossfading, multi-track composition building, and download capabilities.
- **Developer Transparency Mode:** Optional toggle to inspect raw model feature outputs and scoring breakdowns behind candidate selection.

---

## 🛠️ Prerequisites

Before running LISTENER, ensure you have the following installed on your system:

- **Python:** Version `3.10` or higher (Python 3.11 or 3.12 recommended)
- **FFmpeg:** Required for audio reading/processing by `librosa` and PyTorch.
  - **macOS (via Homebrew):** `brew install ffmpeg`
  - **Ubuntu/Debian:** `sudo apt install ffmpeg`
  - **Windows (via Chocolatey):** `choco install ffmpeg`

---

## 🚀 Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Resonate-Data-Lab/ensemble_ML_audio_listener.git
cd ensemble_ML_audio_listener
```

### 2. Set Up a Virtual Environment

It is recommended to use a virtual environment:

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate

# On Windows (Command Prompt):
# .venv\Scripts\activate.bat

# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install all required Python packages using `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** On first run, PANNs (`panns-inference`) will automatically download pre-trained model weights (`Cnn14_mAP=0.431.pth`) to your home folder (`~/panns_data/`).

---

## 🖥️ Running the Application

To launch the LISTENER web application, execute:

```bash
streamlit run app.py
```

Once started, open your web browser and navigate to:
```
http://localhost:8501
```

---

## 📖 How to Use LISTENER

1. **Upload Recording:**
   - Upload any environmental audio file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).
2. **Analyze Recording:**
   - Click **Analyze Recording**. LISTENER analyzes candidate windows across your audio file using signal distinctiveness, layering, and temporal novelty metrics.
3. **Explore & Select Moments:**
   - Listen to each candidate clip surfaced by the AI.
   - Choose which clips to **Keep** or **Discard**.
4. **Apply Affective Transformations (Optional):**
   - Transform selected clips using target affective tones (*Calm*, *Nostalgic*, *Energetic*, etc.).
5. **Edit & Sequence Composition:**
   - Open the **Composition Editor** to arrange your saved clips.
   - Adjust trim start/end times, reorder clips, or configure crossfades between moments.
6. **Export & Download:**
   - Listen to your completed composition and download the final combined audio file along with your research/decision log snapshot.

---

## 📁 Repository Structure

```
ensemble_ML_audio_listener/
├── app.py                  # Main Streamlit web application interface
├── requirements.txt        # Python dependency specifications
├── README.md               # Project documentation
├── listener/               # Core Python library
│   ├── analysis.py         # Signal analysis, windowing, and scoring logic
│   ├── affect.py           # Affective tone transformation functions
│   ├── audio_classifier.py # PANNs CNN model loading & embedding extraction
│   ├── audio_io.py         # Audio file loading, saving, and timestamp formatting
│   ├── clipping.py         # Audio clip extraction and trimming utilities
│   ├── composition.py      # Multi-clip composition rendering & crossfading
│   ├── pipeline.py         # Verification and candidate generation pipeline
│   ├── research_log.py     # Logging decision history and session snapshots
│   └── selection.py        # Candidate selection and re-ranking algorithms
└── data/                   # Temporary directory for saved clips & compositions
```

---

## 🔑 Optional Configuration

- **API Key for Description Generation (Optional):**
  If you have an Anthropic API Key, you can enable auto-generated textual descriptions by setting:
  ```bash
  export ANTHROPIC_API_KEY="your-api-key-here"
  ```

---

## 📄 License & Attribution

Developed by the **Resonate Data Lab** team. Designed for interactive AI-human audio research, sound exploration, and affective acoustic analysis.
