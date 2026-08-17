"""Version 5: user-directed emotional/affective sound transformation.

The person picks an affective DIRECTION; the system transforms the ALREADY
selected candidate clip according to that direction. This module never
analyzes audio to infer an emotion and never claims a clip "is" or "sounds
like" a feeling -- the label is the user's own chosen interpretation, not a
system judgment. This is the same "AI surfaces/transforms, humans interpret"
principle the rest of the app follows, applied to transformation instead of
selection: the system picks HOW to render a chosen direction, never WHICH
direction a sound "really" has.

Each of the 12 tones below uses its OWN combination of techniques -- not the
same processing chain with different numbers. Pitch is deliberately used
sparingly (only Dreamy's "drift" and Uneasy's "instability" lean on it, and
even there only as one ingredient among several) -- the real differentiators
are dynamics (attack/release shape how transients feel), a proper dry/wet
"space send" (asplit + delay + filtering + amix, not just an echo bolted onto
the end of every chain) for distance/proximity, saturation and lofi texture
for warmth/imperfection, amplitude modulation (tremolo) for instability
distinct from pitch modulation (vibrato), selective EQ notches for
"obscured" character, and -- for the tones whose identity is really about
timing rather than spectral shape (Tense, Uneasy, Lonely, Melancholic) -- a
numpy-based post-process that gently recedes a few segments toward (not all
the way to) silence, with soft edges, to create real gaps/separation.

"Try another interpretation" reseeds the SAME recipe's parameter ranges
rather than picking a new recipe, so every version of a direction stays
recognizably in that direction while still varying.

Transformations always run on the already-exported candidate clip (never the
original recording, never regenerated audio) and are written to their own
output file -- the source clip is never modified in place.
"""
import itertools
import random
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

TRANSFORM_DIR = Path(__file__).resolve().parent.parent / "data" / "transforms"


def _jitter_mult(rng: random.Random, nominal: float, spread: float) -> float:
    """nominal scaled by a random factor within +/-spread (e.g. spread=0.1 -> +/-10%)."""
    return nominal * (1.0 + rng.uniform(-spread, spread))


def _jitter_add(rng: random.Random, nominal: float, spread: float) -> float:
    """nominal offset by a random amount within +/-spread, in the parameter's own units."""
    return nominal + rng.uniform(-spread, spread)


@dataclass
class Tone:
    key: str  # stable identifier, used in session state and filenames
    label: str  # display name, e.g. "Nostalgic"
    descriptors: str  # the short qualifying phrase, e.g. "warm, distant, memory-like"
    recipe: object  # callable(rng: random.Random) -> dict of DSP parameters


# ---------------------------------------------------------------------------
# Group 1: introspective/soft directions. Each gets a DIFFERENT mechanism for
# "softness" so they stop collapsing into the same slow+dark+quiet result:
#   Calm        -- minimal processing, just smoothed (barely any space send)
#   Nostalgic   -- warmth (saturation) + texture (lofi) + volume instability
#   Melancholic -- slowest tempo + darkest EQ + longest single sustained dip
#   Mysterious  -- selective EQ notch (obscures) + a single filtered "ghost" repeat
#   Dreamy      -- pitch DRIFT + tall wet reverb tail + dual-mechanism wide stereo
#   Lonely      -- longest single distant repeat + narrowed field + explicit gaps
#   Intimate    -- NO space send at all (fully dry) + presence boost + narrow-close
#   Serene      -- heaviest/slowest compression (stillness) + tall wet tail + wide
# ---------------------------------------------------------------------------


def _calm(rng):
    return {
        "tempo": _jitter_mult(rng, 0.98, 0.015),
        "lowpass_hz": _jitter_mult(rng, 8500, 0.08),
        "compressor": (0.20, _jitter_add(rng, 2.5, 0.4), _jitter_add(rng, 8, 3), 180),
        "space": (_jitter_add(rng, 35, 8), 7000, _jitter_add(rng, -14, 2), _jitter_add(rng, 0.12, 0.04), None),
        "volume_db": _jitter_add(rng, -0.5, 0.3),
    }


def _nostalgic(rng):
    return {
        "tempo": _jitter_mult(rng, 0.98, 0.015),
        "lowpass_hz": _jitter_mult(rng, 4200, 0.1),
        "eq_bands": [(_jitter_add(rng, 250, 30), _jitter_add(rng, 2.5, 0.6), 1.2)],
        "saturation": (_jitter_add(rng, 5.0, 1.0), _jitter_add(rng, 0.75, 0.08)),
        "texture": (_jitter_add(rng, 7.0, 1.0), _jitter_add(rng, 0.2, 0.06)),
        "tremolo": (_jitter_add(rng, 1.2, 0.3), _jitter_add(rng, 0.15, 0.05)),
        "vibrato": (_jitter_add(rng, 0.6, 0.15), _jitter_add(rng, 0.06, 0.02)),
        "space": (_jitter_add(rng, 120, 20), 3200, _jitter_add(rng, -12, 2), _jitter_add(rng, 0.28, 0.06), (0.6, 0.5, 90, 0.25)),
        "stereo_width": _jitter_mult(rng, 0.75, 0.08),
        "volume_db": _jitter_add(rng, -1.5, 0.4),
    }


def _melancholic(rng):
    return {
        "tempo": _jitter_mult(rng, 0.90, 0.02),
        "lowpass_hz": _jitter_mult(rng, 3600, 0.1),
        "compressor": (0.15, _jitter_add(rng, 3.5, 0.5), _jitter_add(rng, 35, 8), 350),
        "space": (_jitter_add(rng, 150, 25), 2800, _jitter_add(rng, -10, 2), _jitter_add(rng, 0.32, 0.06), (0.6, 0.55, 130, 0.3)),
        "stereo_width": _jitter_mult(rng, 0.8, 0.08),
        "volume_db": _jitter_add(rng, -2.0, 0.4),
        "gating": {"segment_seconds": 0.9, "count_range": (1, 1), "depth_db": -10, "fade_ms": 150, "jitter_position": False},
    }


def _mysterious(rng):
    return {
        "tempo": _jitter_mult(rng, 0.97, 0.015),
        "eq_bands": [
            (_jitter_add(rng, 1400, 200), _jitter_add(rng, -5.5, 1.0), 1.4),
            (_jitter_add(rng, 6500, 500), _jitter_add(rng, -4.0, 1.0), 1.2),
        ],
        "vibrato": (_jitter_add(rng, 0.4, 0.1), _jitter_add(rng, 0.05, 0.02)),
        "space": (_jitter_add(rng, 220, 30), 2600, _jitter_add(rng, -13, 2), _jitter_add(rng, 0.3, 0.05), None),
        "stereo_width": _jitter_mult(rng, 1.15, 0.1),
        "volume_db": _jitter_add(rng, -1.5, 0.4),
    }


def _dreamy(rng):
    return {
        "tempo": _jitter_mult(rng, 0.96, 0.015),
        "lowpass_hz": _jitter_mult(rng, 4800, 0.1),
        "compressor": (0.15, 3.0, _jitter_add(rng, 6, 2), 200),
        "vibrato": (_jitter_add(rng, 2.2, 0.4), _jitter_add(rng, 0.22, 0.05)),
        "space": (_jitter_add(rng, 180, 25), 3600, _jitter_add(rng, -8, 1.5), _jitter_add(rng, 0.42, 0.06), (0.7, 0.65, 150, 0.4)),
        "haas": True,
        "stereo_width": _jitter_mult(rng, 1.5, 0.1),
        "volume_db": _jitter_add(rng, -1.0, 0.4),
    }


def _lonely(rng):
    return {
        "tempo": _jitter_mult(rng, 0.97, 0.015),
        "lowpass_hz": _jitter_mult(rng, 3400, 0.1),
        "space": (_jitter_add(rng, 320, 40), 2200, _jitter_add(rng, -11, 2), _jitter_add(rng, 0.35, 0.06), None),
        "stereo_width": _jitter_mult(rng, 0.55, 0.08),
        "volume_db": _jitter_add(rng, -3.5, 0.5),
        "gating": {"segment_seconds": 0.7, "count_range": (1, 2), "depth_db": -22, "fade_ms": 100, "jitter_position": False},
    }


def _intimate(rng):
    return {
        "tempo": _jitter_mult(rng, 1.0, 0.005),
        "highpass_hz": _jitter_mult(rng, 90, 0.2),
        "eq_bands": [(_jitter_add(rng, 3200, 300), _jitter_add(rng, 3.5, 0.8), 1.1)],
        "exciter": (_jitter_add(rng, 1.6, 0.3), 4500),
        "compressor": (0.22, _jitter_add(rng, 2.2, 0.4), 12, 100),
        "stereo_width": _jitter_mult(rng, 0.55, 0.08),
        "volume_db": _jitter_add(rng, 0.5, 0.3),
    }


def _serene(rng):
    return {
        "tempo": _jitter_mult(rng, 0.95, 0.01),
        "lowpass_hz": _jitter_mult(rng, 6000, 0.08),
        "compressor": (0.10, _jitter_add(rng, 4.5, 0.6), _jitter_add(rng, 40, 8), 400),
        "space": (_jitter_add(rng, 90, 15), 4200, _jitter_add(rng, -6, 1), _jitter_add(rng, 0.45, 0.06), (0.65, 0.6, 140, 0.35)),
        "haas": True,
        "stereo_width": _jitter_mult(rng, 1.4, 0.1),
        "volume_db": _jitter_add(rng, -1.5, 0.3),
    }


# ---------------------------------------------------------------------------
# Group 2: bright/active directions. Each gets a different mechanism so they
# stop collapsing into "higher pitch + more energy":
#   Joyful    -- brightness + slow-attack transient definition + rhythmic
#                tremolo "bounce" -- fully dry (immediate/present)
#   Energetic -- short (<40ms) "doubling" delay for density/movement + the
#                heaviest fast-attack pumping compression + fastest tempo
#   Tense     -- hardest/fastest compression (pressured) + a single tight
#                dry slap-delay ("controlled repetition") + abrupt regular gaps
#   Uneasy    -- pitch instability (vibrato) + amplitude instability (tremolo)
#                + irregular jittered gaps, all layered together -- no other
#                tone combines three kinds of instability at once
# ---------------------------------------------------------------------------


def _joyful(rng):
    return {
        "tempo": _jitter_mult(rng, 1.04, 0.015),
        "eq_bands": [(_jitter_add(rng, 5500, 400), _jitter_add(rng, 3.5, 0.8), 1.0)],
        "exciter": (_jitter_add(rng, 1.2, 0.3), 6000),
        "compressor": (0.25, 1.8, _jitter_add(rng, 20, 5), 60),
        "tremolo": (_jitter_add(rng, 3.5, 0.6), _jitter_add(rng, 0.18, 0.05)),
        "stereo_width": _jitter_mult(rng, 1.35, 0.1),
        "volume_db": _jitter_add(rng, 0.5, 0.4),
    }


def _energetic(rng):
    return {
        "tempo": _jitter_mult(rng, 1.09, 0.015),
        # a punchy midrange boost, deliberately far from Joyful's ~5500Hz "air" band
        # and Tense's ~3400Hz "edge" band, so the three don't converge on the same
        # brightness -- Energetic's character is fullness/punch, not shimmer or edge.
        "eq_bands": [(_jitter_add(rng, 1900, 250), _jitter_add(rng, 3.5, 0.8), 1.1)],
        "compressor": (0.12, _jitter_add(rng, 5.0, 0.8), _jitter_add(rng, 3, 1), 45),
        # short "doubling" delay -- a chorus-like density/thickening, not a distance
        # effect (the wet copy stays bright, lowpass=9000) -- unique to this tone
        "space": (_jitter_add(rng, 18, 4), 9000, _jitter_add(rng, -4, 1), _jitter_add(rng, 0.4, 0.05), None),
        "stereo_width": _jitter_mult(rng, 1.3, 0.1),
        "volume_db": _jitter_add(rng, 1.0, 0.4),
    }


def _tense(rng):
    return {
        "tempo": _jitter_mult(rng, 1.015, 0.01),
        "highpass_hz": _jitter_mult(rng, 350, 0.15),
        # a narrow, higher edge/sharpness band -- distinct from Energetic's punchy
        # ~1900Hz fullness and Joyful's ~5500Hz air, so Tense reads as "thin and
        # hard-edged" (helped by the highpass removing warmth) rather than "bright".
        "eq_bands": [(_jitter_add(rng, 3400, 300), _jitter_add(rng, 2.5, 0.6), 1.4)],
        "compressor": (0.06, _jitter_add(rng, 7.0, 1.0), _jitter_add(rng, 2, 1), 30),
        "space": (_jitter_add(rng, 45, 10), 3500, _jitter_add(rng, -16, 2), _jitter_add(rng, 0.1, 0.03), (0.5, 0.4, 55, 0.15)),
        "volume_db": _jitter_add(rng, 0.5, 0.3),
        "gating": {"segment_seconds": 0.4, "count_range": (1, 2), "depth_db": -20, "fade_ms": 15, "jitter_position": False},
    }


def _uneasy(rng):
    return {
        "tempo": _jitter_mult(rng, 1.0, 0.02),
        "eq_bands": [(_jitter_add(rng, 1100, 200), _jitter_add(rng, -3.0, 1.0), 1.3)],
        "vibrato": (_jitter_add(rng, 4.5, 1.0), _jitter_add(rng, 0.13, 0.04)),
        "tremolo": (_jitter_add(rng, 0.8, 0.2), _jitter_add(rng, 0.12, 0.04)),
        "stereo_width": _jitter_mult(rng, 1.2, 0.15),
        "volume_db": _jitter_add(rng, 0.0, 0.3),
        "gating": {"segment_seconds": 0.5, "count_range": (1, 3), "depth_db": -14, "fade_ms": 70, "jitter_position": True},
    }


TONES = [
    Tone("calm", "Calm", "peaceful, soft, settled", _calm),
    Tone("nostalgic", "Nostalgic", "warm, distant, memory-like", _nostalgic),
    Tone("joyful", "Joyful", "lively, playful, bright", _joyful),
    Tone("melancholic", "Melancholic", "reflective, wistful, subdued", _melancholic),
    Tone("tense", "Tense", "pressured, uneasy, anticipatory", _tense),
    Tone("mysterious", "Mysterious", "ambiguous, unfamiliar, intriguing", _mysterious),
    Tone("dreamy", "Dreamy", "hazy, floating, unreal", _dreamy),
    Tone("lonely", "Lonely", "isolated, distant, spacious", _lonely),
    Tone("intimate", "Intimate", "close, personal, delicate", _intimate),
    Tone("energetic", "Energetic", "active, lively, stimulating", _energetic),
    Tone("uneasy", "Uneasy", "strange, unstable, uncomfortable", _uneasy),
    Tone("serene", "Serene", "still, tranquil, deeply peaceful", _serene),
]
TONES_BY_KEY = {t.key: t for t in TONES}


def _probe_sample_rate(filepath: str) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate", "-of", "csv=p=0", filepath,
        ],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def _build_filter_graph(p: dict, sample_rate: int) -> str:
    """Build a full ffmpeg filter_complex graph for one recipe -- not a single
    linear chain reused by every tone. The "space" parameter, when present,
    genuinely branches the signal (asplit), processes a wet copy separately
    (delay + filtering + optional echo tail), and mixes it back with the dry
    signal at a controlled ratio (amix) -- this is what makes distance/
    proximity a real, continuously-controllable dimension instead of just
    "echo present or not". Always processes as stereo (aformat first) so
    stereo-width tools work regardless of the source clip's channel count.
    Always ends with a limiter so no combination of boosts can clip.
    """
    fragments = []
    counter = itertools.count()

    def new_label():
        return f"n{next(counter)}"

    def chain(cur, filt):
        nxt = new_label()
        fragments.append(f"[{cur}]{filt}[{nxt}]")
        return nxt

    cur = chain("0:a", "aformat=channel_layouts=stereo")

    pitch_semitones = p.get("pitch_semitones", 0.0)
    if abs(pitch_semitones) > 0.01:
        ratio = 2 ** (pitch_semitones / 12.0)
        new_rate = max(1000, int(round(sample_rate * ratio)))
        cur = chain(cur, f"asetrate={new_rate}")
        cur = chain(cur, f"aresample={sample_rate}")
        tempo_compensation = 1.0 / ratio
    else:
        tempo_compensation = 1.0
    combined_tempo = p.get("tempo", 1.0) * tempo_compensation
    if abs(combined_tempo - 1.0) > 0.005:
        cur = chain(cur, f"atempo={combined_tempo:.4f}")

    if p.get("highpass_hz"):
        cur = chain(cur, f"highpass=f={p['highpass_hz']:.0f}")
    if p.get("lowpass_hz"):
        cur = chain(cur, f"lowpass=f={p['lowpass_hz']:.0f}")
    for freq, gain, width in p.get("eq_bands", []):
        cur = chain(cur, f"equalizer=f={freq:.0f}:t=o:w={width}:g={gain:.2f}")

    if p.get("saturation"):
        drive, threshold = p["saturation"]
        cur = chain(cur, f"volume={drive:.2f}dB")
        cur = chain(cur, f"asoftclip=type=tanh:threshold={min(max(threshold, 0.05), 1.0):.3f}")
        cur = chain(cur, f"volume={-drive * 0.6:.2f}dB")

    if p.get("texture"):
        bits, mix = p["texture"]
        cur = chain(cur, f"acrusher=bits={min(max(bits, 4), 16):.1f}:mix={min(max(mix, 0.0), 1.0):.2f}:samples=2")

    if p.get("exciter"):
        amount, freq = p["exciter"]
        cur = chain(cur, f"aexciter=amount={min(max(amount, 0.1), 8.0):.2f}:freq={freq:.0f}")

    if p.get("vibrato"):
        freq, depth = p["vibrato"]
        cur = chain(cur, f"vibrato=f={max(0.1, freq):.2f}:d={min(max(depth, 0.0), 1.0):.3f}")
    if p.get("tremolo"):
        freq, depth = p["tremolo"]
        cur = chain(cur, f"tremolo=f={max(0.1, freq):.2f}:d={min(max(depth, 0.0), 1.0):.3f}")

    if p.get("compressor"):
        threshold, ratio, attack, release = p["compressor"]
        cur = chain(
            cur,
            f"acompressor=threshold={min(max(threshold, 0.01), 1.0):.3f}:"
            f"ratio={min(max(ratio, 1.0), 20.0):.2f}:attack={max(1, attack):.0f}:release={max(1, release):.0f}",
        )

    if p.get("space"):
        delay_ms, wet_lowpass_hz, decay_db, wet_ratio, tail_echo = p["space"]
        dry_label = new_label()
        wet_label = new_label()
        fragments.append(f"[{cur}]asplit=2[{dry_label}][{wet_label}]")
        w = chain(wet_label, f"adelay={max(1, delay_ms):.0f}:all=1")
        w = chain(w, f"lowpass=f={wet_lowpass_hz:.0f}")
        w = chain(w, f"volume={decay_db:.2f}dB")
        if tail_echo:
            in_gain, out_gain, echo_delay, echo_decay = tail_echo
            w = chain(
                w,
                f"aecho={min(max(in_gain, 0.0), 1.0):.2f}:{min(max(out_gain, 0.0), 1.0):.2f}:"
                f"{max(1, echo_delay):.0f}:{min(max(echo_decay, 0.0), 0.95):.3f}",
            )
        wet_ratio = min(max(wet_ratio, 0.0), 0.9)
        mixed = new_label()
        fragments.append(
            f"[{dry_label}][{w}]amix=inputs=2:weights='{1.0 - wet_ratio:.2f} {wet_ratio:.2f}':normalize=0[{mixed}]"
        )
        cur = mixed

    used_haas = bool(p.get("haas"))
    if used_haas:
        cur = chain(cur, "haas")
    width = p.get("stereo_width")
    if width is not None and abs(width - 1.0) > 0.02:
        if width > 1.0 and not used_haas:
            # extrastereo alone only AMPLIFIES an existing L/R difference -- on a
            # genuinely mono-sourced clip (single-mic environmental recordings,
            # the common case) there's nothing for it to amplify, so it's a
            # silent no-op. A small Haas-style delay on the right channel first
            # creates a real, audible inter-channel difference (regardless of
            # source channel count) for extrastereo to then widen further.
            # Skipped when `haas` already ran above -- that already produces a
            # genuine, wide stereo image on its own; stacking a second,
            # independent delay on top of it just adds uncontrolled comb-filter
            # coloration rather than more usable width.
            widen_ms = min((width - 1.0) * 14.0, 18.0)
            cur = chain(cur, f"adelay=0|{widen_ms:.1f}")
            cur = chain(cur, f"extrastereo=m={min(width, 2.5):.2f}")
        elif width <= 1.0:
            # narrowing toward mono is a safe, unconditional operation regardless
            # of source channel content -- extrastereo alone is sufficient here.
            cur = chain(cur, f"extrastereo=m={max(width, 0.0):.2f}")

    volume_db = p.get("volume_db", 0.0)
    if abs(volume_db) > 0.01:
        cur = chain(cur, f"volume={volume_db:.2f}dB")

    fragments.append(f"[{cur}]alimiter=limit=0.97[out]")
    return ";".join(fragments)


def _apply_gating(audio: np.ndarray, sample_rate: int, rng: random.Random, gating: dict) -> np.ndarray:
    """Gently recede a handful of segments toward (not necessarily all the way
    to) silence, with a short fade at each edge -- reads as a natural pause or
    a receded moment in the SAME recording rather than a jump-cut or click.
    Used for tones whose identity depends on timing/density/separation rather
    than spectral or dynamics processing alone (Tense, Uneasy, Lonely,
    Melancholic). The very first and last segment are never touched, so the
    clip always stays recognizable at its edges.
    """
    segment_len = max(1, int(gating["segment_seconds"] * sample_rate))
    n_segments = len(audio) // segment_len
    if n_segments < 3:
        return audio

    lo, hi = gating["count_range"]
    count = min(rng.randint(lo, hi), n_segments - 2)
    if count <= 0:
        return audio

    candidates = list(range(1, n_segments - 1))
    rng.shuffle(candidates)
    chosen = sorted(candidates[:count])

    depth_mult = 10 ** (gating["depth_db"] / 20.0)
    fade_len = max(1, int((gating["fade_ms"] / 1000.0) * sample_rate))
    jitter = gating.get("jitter_position", False)

    out = audio.copy()
    for idx in chosen:
        start = idx * segment_len
        end = min(len(out), start + segment_len)
        if jitter:
            shift = rng.randint(-fade_len, fade_len)
            start = max(0, start + shift)
            end = min(len(out), end + shift)
        seg_len = end - start
        if seg_len <= 2 * fade_len:
            continue
        envelope = np.ones(seg_len)
        envelope[:fade_len] = np.linspace(1.0, depth_mult, fade_len)
        envelope[fade_len: seg_len - fade_len] = depth_mult
        envelope[seg_len - fade_len:] = np.linspace(depth_mult, 1.0, fade_len)
        if out.ndim == 2:
            out[start:end] *= envelope[:, None]
        else:
            out[start:end] *= envelope
    return out


def transform_clip(source_filepath: str, tone_key: str, version: int) -> str:
    """Render one variation of tone_key's transformation applied to the clip at
    source_filepath, writing a new file (the source is never modified). The
    same (source, tone_key, version) always renders identically -- seeded by
    the tone/version, not randomly -- so "Try another interpretation" explores
    a genuinely different but reproducible variation within that tone's range,
    never a coin-flip. Never touches the original recording or re-runs any
    classification; this is a pure audio-effects transform of an
    already-exported clip.
    """
    tone = TONES_BY_KEY[tone_key]
    rng = random.Random(f"{tone_key}:{version}:{Path(source_filepath).name}")
    params = tone.recipe(rng)
    sample_rate = _probe_sample_rate(source_filepath)
    filter_graph = _build_filter_graph(params, sample_rate)

    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)
    rendered_path = TRANSFORM_DIR / f"{uuid.uuid4().hex[:8]}_{tone_key}_v{version}_render.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", source_filepath,
            "-filter_complex", filter_graph, "-map", "[out]",
            "-c:a", "pcm_s16le", str(rendered_path),
        ],
        check=True,
    )

    gating = params.get("gating")
    output_path = TRANSFORM_DIR / f"{uuid.uuid4().hex[:8]}_{tone_key}_v{version}.wav"
    if gating:
        audio, sr = sf.read(str(rendered_path))
        gated = _apply_gating(audio, sr, rng, gating)
        sf.write(str(output_path), gated, sr, subtype="PCM_16")
        rendered_path.unlink(missing_ok=True)
    else:
        rendered_path.rename(output_path)

    return str(output_path)
