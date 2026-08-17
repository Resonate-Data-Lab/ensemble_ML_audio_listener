"""Step 8: conservative, classifier-only sound labels for each candidate.

Turns a candidate's raw AudioSet labels into a short, honest label line and
a list of detected-sound labels, each keeping both the raw AudioSet class
name (for research traceability) and a cleaned-up display form (for
readability).

This module NEVER assembles labels into a sentence and NEVER uses an LLM to
interpret, narrate, or infer a relationship between them (spec sections
12-14, and Version 4: PANNs is a sound-event tagger, not a scene-understanding
model -- joining labels with "with"/"and" reads as a claim about how the
sounds relate that the classifier never made). If multiple labels are
confidently detected, they're shown side by side (e.g. "Water · Rain"), never
fused into a fabricated description. If nothing clears the display-confidence
threshold, the label line says so plainly ("Unclear sound") instead of
guessing.

candidate.labels is expected to already reflect the FINAL EXPORTED CLIP
(see pipeline.generate_verified_candidates, which re-classifies each clip
after its boundaries are finalized), not the original analysis window --
this module just turns whatever labels it's given into text, it doesn't
decide which audio they came from.
"""
import re
from dataclasses import dataclass, field

from .selection import Candidate

MAX_LABELS = 4  # a bit richer than 3: show more of what's confidently, concurrently present
LABEL_SCORE_THRESHOLD_FOR_DISPLAY = 0.12
LABEL_SEPARATOR = " · "
UNCLEAR_LABEL_TEXT = "Unclear sound"

# A trailing parenthetical on an AudioSet label is a disambiguator AudioSet adds
# for words that appear more than once in its ontology (e.g. "Zipper (clothing)",
# "Rattle (instrument)") -- it exists to keep classes distinct internally, not to
# add descriptive meaning, so it's dropped for display. Two of AudioSet's 527
# classes happen to strip down to a name that collides with a different, unrelated
# bare class ("Whimper" / "Whimper (dog)" and "Rattle" / "Rattle (instrument)") --
# raw_label below always keeps the exact, unambiguous AudioSet class name so that
# collision never affects traceability, only the shortened display text.
_TRAILING_QUALIFIER_RE = re.compile(r"^(?P<base>.*\S)\s*\([^()]*\)\s*$")


@dataclass
class DetectedSound:
    raw_label: str  # exact AudioSet class name, unmodified -- for traceability
    display_label: str  # human-readable form shown in the UI
    confidence: float  # classifier score for raw_label, 0-1


@dataclass
class Description:
    text: str
    detected_sounds: list = field(default_factory=list)  # list[DetectedSound]


def _humanize_label(label: str) -> str:
    """Clean up a raw AudioSet class name for display without discarding its
    specificity. AudioSet already writes many labels as a full descriptive
    phrase (e.g. "Dishes, pots, and pans", "Bird vocalization, bird call, bird
    song") -- that's kept in full, not truncated at the first comma, since each
    term is additional real specificity rather than a redundant synonym to drop.
    The only thing removed is a trailing parenthetical disambiguator (see
    _TRAILING_QUALIFIER_RE above) -- everything else is just a lowercase-first
    pass so labels read as ordinary text instead of title-cased class names.
    """
    match = _TRAILING_QUALIFIER_RE.match(label)
    text = match.group("base") if match else label
    if text.isupper() or len(text) <= 3:
        return text
    return text[0].lower() + text[1:]


def _capitalize_first(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _selected_labels(candidate: Candidate, limit: int) -> list:
    """Return up to `limit` DetectedSound entries, strongest first, deduplicated
    by display text (distinct AudioSet classes can rarely humanize to the same
    display string -- see _TRAILING_QUALIFIER_RE)."""
    seen = set()
    result = []
    for raw_label, score in candidate.labels:
        if score < LABEL_SCORE_THRESHOLD_FOR_DISPLAY:
            continue
        display_label = _humanize_label(raw_label)
        if display_label in seen:
            continue
        seen.add(display_label)
        result.append(DetectedSound(raw_label=raw_label, display_label=display_label, confidence=score))
        if len(result) >= limit:
            break
    return result


def _label_line(detected: list) -> str:
    """Build the short, honest label line shown to the user: a single label, or
    several confidently-detected labels shown side by side (never fused into a
    sentence -- see module docstring). If nothing cleared the display-confidence
    threshold, says so plainly rather than inventing a description."""
    if not detected:
        return UNCLEAR_LABEL_TEXT
    return LABEL_SEPARATOR.join(_capitalize_first(d.display_label) for d in detected)


def describe_candidate(candidate: Candidate) -> Description:
    detected = _selected_labels(candidate, MAX_LABELS)
    return Description(text=_label_line(detected), detected_sounds=detected)
