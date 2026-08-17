"""Step 6: build a candidate pool, then a diversity-aware shortlist from it.

Two stages, matching the brief's requested architecture:

    windows -> build_candidate_pool()   -- every eligible window, scored
            -> select_diverse_candidates() -- a diverse shortlist from that pool

Each pooled window gets five 0-1 component scores:

  A/F. clarity   -- confidence of the window's top label (a clearly-heard,
                     confidently-classified event, not a guess)
  B.   novelty    -- embedding distance from the previous moment (something
                     changed here: entered, left, or shifted)
  C.   rarity     -- inverse frequency of the window's top label across the
                     whole recording (infrequent events score higher)
  D.   layering   -- how many labels were concurrently active
  E.   contrast   -- embedding distance from the local (~30s) neighborhood
                     average (stands out from its surroundings)

select_diverse_candidates() combines these into one composite score, then
greedily builds a shortlist: at each step it picks the pooled window with the
highest score *after* two soft diversity penalties (has this label already
been picked? is this acoustically close to something already picked?), on
top of two hard guards (minimum time separation, and a near-duplicate
embedding cutoff that always applies regardless of penalties). This is a
maximal-marginal-relevance-style selection -- diversity is part of what's
being optimized, not a filter applied after the fact to a plain top-N list.

Optional weight jitter + score jitter (both seeded, so a given seed always
reproduces the same shortlist) let the same pool surface a *different* valid
shortlist on repeat runs -- see listener.pipeline for how "Analyze Again"
uses this without re-running the classifier. Windows with any detected speech
are excluded entirely, before any of this scoring happens. This module only
ranks and selects; it does not judge meaning or extract audio (Step 7) or
describe it in language (Step 8).
"""
import random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .analysis import WindowAnalysis

MAX_CANDIDATES = 10
MIN_SEPARATION_SECONDS = 8.0  # keeps final ~5s clips from overlapping or feeling redundant
CONTRAST_WINDOW_SECONDS = 30.0  # how far around a moment counts as its "surroundings"
HARD_DUPLICATE_SIMILARITY = 0.97  # skip outright: this close in embedding space is a near-literal repeat

CATEGORY_DIVERSITY_WEIGHT = 0.15  # soft penalty per already-selected candidate sharing this top label
SIMILARITY_DIVERSITY_WEIGHT = 0.20  # soft penalty scaled by closeness to the nearest already-selected candidate

# Fixed (not scaled) penalty for a window that was part of a PREVIOUS run's shown
# results (see previously_selected_starts below). Deliberately larger than any
# plausible composite-score gap or jitter swing so a previously-surfaced window
# reliably loses to a fresh alternative when one exists, rather than merely
# reshuffling among near-ties the way weight/score jitter alone does. Still a
# soft penalty, not an exclusion: if nothing better remains, the same window can
# still win -- "Analyze Again" never fabricates variation that isn't there.
PREVIOUS_RUN_PENALTY = 0.5

DEFAULT_WEIGHTS = {
    "novelty": 0.25,
    "contrast": 0.20,
    "rarity": 0.20,
    "layering": 0.15,
    "clarity": 0.20,
}

JITTER_WEIGHT_RANGE = 0.40  # +/- 40% per-weight perturbation when a jitter_seed is given
JITTER_SCORE_RANGE = 0.15  # +/- 15% per-candidate composite-score perturbation


@dataclass
class Candidate:
    start_seconds: float
    labels: list  # [(label, score), ...], same shape as WindowAnalysis.labels
    score: float  # final composite score used for this selection -- a ranking signal, not an importance judgment
    scores: dict = field(default_factory=dict)  # component breakdown (novelty/contrast/rarity/layering/clarity/...)


@dataclass
class _PooledWindow:
    window: WindowAnalysis
    scores: dict  # novelty/contrast/rarity/layering/clarity, each normalized 0-1


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _min_max_normalize(values: list) -> list:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _top_label(window: WindowAnalysis):
    return window.labels[0][0] if window.labels else None


def _novelty_scores(eligible: list) -> list:
    """B. Distance from each window's embedding to the previous one in time."""
    scores = [0.0]
    for i in range(1, len(eligible)):
        scores.append(1.0 - _cosine_similarity(eligible[i].embedding, eligible[i - 1].embedding))
    return scores


def _contrast_scores(eligible: list) -> list:
    """E. Distance from each window's embedding to its local neighborhood average.

    Uses a sliding window (eligible is time-sorted) instead of comparing every
    pair, so this stays fast on a 30-45 minute recording.
    """
    starts = [w.start_seconds for w in eligible]
    embeddings = [w.embedding for w in eligible]
    n = len(eligible)
    scores = [0.0] * n
    left = right = 0
    for i in range(n):
        while starts[i] - starts[left] > CONTRAST_WINDOW_SECONDS:
            left += 1
        if right < i:
            right = i
        while right + 1 < n and starts[right + 1] - starts[i] <= CONTRAST_WINDOW_SECONDS:
            right += 1
        neighbor_idxs = [j for j in range(left, right + 1) if j != i]
        if neighbor_idxs:
            local_mean = np.mean([embeddings[j] for j in neighbor_idxs], axis=0)
            scores[i] = 1.0 - _cosine_similarity(embeddings[i], local_mean)
    return scores


def build_candidate_pool(windows: list[WindowAnalysis]) -> list[_PooledWindow]:
    """Score every eligible window (has a label, no speech detected anywhere in
    it). This is the full candidate pool, deliberately much larger than the
    final ~10 -- select_diverse_candidates() draws the shortlist from it."""
    eligible = [w for w in windows if w.labels and not w.has_speech]
    if not eligible:
        return []

    label_counts = Counter(_top_label(w) for w in eligible)

    novelty = _min_max_normalize(_novelty_scores(eligible))
    contrast = _min_max_normalize(_contrast_scores(eligible))
    rarity = _min_max_normalize([1.0 / label_counts[_top_label(w)] for w in eligible])
    layering = _min_max_normalize([float(len(w.labels)) for w in eligible])
    clarity = _min_max_normalize([w.labels[0][1] for w in eligible])

    return [
        _PooledWindow(
            window=w,
            scores={
                "novelty": novelty[i],
                "contrast": contrast[i],
                "rarity": rarity[i],
                "layering": layering[i],
                "clarity": clarity[i],
            },
        )
        for i, w in enumerate(eligible)
    ]


def select_diverse_candidates(
    pool: list[_PooledWindow],
    count: int = MAX_CANDIDATES,
    weights: dict = None,
    jitter_seed: int = None,
    previously_selected_starts: set = None,
) -> list[Candidate]:
    """Greedily build a diverse shortlist of up to `count` candidates from the pool.

    weights: override DEFAULT_WEIGHTS (must cover the same five component keys).
    jitter_seed: if given, deterministically perturbs both the weights and each
    candidate's composite score by a bounded random amount, so the same pool can
    surface a different (but still quality-gated) shortlist on repeat runs --
    see listener.pipeline / app.py's "Analyze Again". Leave as None for the
    canonical, unjittered ranking.
    previously_selected_starts: start_seconds of windows already SHOWN to the
    user in an earlier run on this same recording (see app.py). Each gets
    PREVIOUS_RUN_PENALTY subtracted during selection, so "Analyze Again"
    actively steers toward moments not shown before rather than relying on
    jitter alone to reshuffle the same top candidates. Windows match exactly
    because they come from the same cached WindowAnalysis list across runs
    (see app.py's `windows` session-state caching), not a recomputed value.

    Returns candidates in chronological order. Returns fewer than `count` if
    the pool doesn't have that many genuinely diverse moments -- candidates
    are never fabricated to reach the target count.
    """
    if not pool:
        return []

    weights = dict(weights or DEFAULT_WEIGHTS)
    rng = random.Random(jitter_seed) if jitter_seed is not None else None

    if rng is not None:
        for key in weights:
            weights[key] *= 1.0 + rng.uniform(-JITTER_WEIGHT_RANGE, JITTER_WEIGHT_RANGE)
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}

    base_scores = []
    for pw in pool:
        composite = sum(weights[key] * pw.scores[key] for key in DEFAULT_WEIGHTS)
        if rng is not None:
            composite *= 1.0 + rng.uniform(-JITTER_SCORE_RANGE, JITTER_SCORE_RANGE)
        base_scores.append(composite)

    remaining = list(range(len(pool)))
    selected: list[tuple] = []  # (composite_score, _PooledWindow)
    category_counts: Counter = Counter()

    while len(selected) < count and remaining:
        best_idx = None
        best_value = -float("inf")

        for idx in remaining:
            pw = pool[idx]
            w = pw.window

            if any(abs(w.start_seconds - s.window.start_seconds) < MIN_SEPARATION_SECONDS for _, s in selected):
                continue
            if any(_cosine_similarity(w.embedding, s.window.embedding) >= HARD_DUPLICATE_SIMILARITY for _, s in selected):
                continue

            value = base_scores[idx]
            value -= CATEGORY_DIVERSITY_WEIGHT * category_counts[_top_label(w)]
            if selected:
                max_similarity = max(_cosine_similarity(w.embedding, s.window.embedding) for _, s in selected)
                value -= SIMILARITY_DIVERSITY_WEIGHT * max_similarity
            if previously_selected_starts and w.start_seconds in previously_selected_starts:
                value -= PREVIOUS_RUN_PENALTY

            if value > best_value:
                best_value = value
                best_idx = idx

        if best_idx is None:
            break  # everything left violates a hard guard

        pw = pool[best_idx]
        selected.append((base_scores[best_idx], pw))
        category_counts[_top_label(pw.window)] += 1
        remaining.remove(best_idx)

    selected.sort(key=lambda pair: pair[1].window.start_seconds)  # chronological display order

    return [
        Candidate(
            start_seconds=pw.window.start_seconds,
            labels=pw.window.labels,
            score=score,
            scores=dict(pw.scores),
        )
        for score, pw in selected
    ]


def select_candidates(windows: list[WindowAnalysis], max_candidates: int = MAX_CANDIDATES) -> list[Candidate]:
    """Convenience wrapper: build the pool and select a shortlist in one call."""
    return select_diverse_candidates(build_candidate_pool(windows), count=max_candidates)
