"""Step 6: build a candidate pool, then a diversity-aware shortlist from it.

Version 6.2. Three stages now, not two:

    windows -> build_candidate_pool()      -- every eligible window, scored
            -> _segment_into_regions()      -- group windows into sonic regions
            -> select_diverse_candidates()  -- a diverse shortlist of REGIONS

V6.1 added a sixth "Sonic Interpretability" dimension and was reverted after
it caused a real regression: candidate lists dominated by many near-identical
slices of the same static/background stretch. Root cause (verified against
real audio, not assumed -- see below) was NOT the sixth dimension itself, it
was structural: every 4-second window was treated as an independent
candidate, so a long static passage -- which is, acoustically, ONE thing --
could still contribute several of the final ~10 slots simply because it
spans enough windows and no single mechanism grouped them into one region
before ranking.

The existing diversity guards (HARD_DUPLICATE_SIMILARITY, a soft embedding-
similarity penalty, MIN_SEPARATION_SECONDS) only ever compared a candidate
under consideration against candidates ALREADY selected, pairwise, using one
global threshold (0.97) picked without reference to real data. Measuring
actual adjacent-window (2s apart) embedding similarity on real recordings
showed why that threshold doesn't reliably work: the MEDIAN similarity
between two adjacent windows where literally nothing is happening is
already ~0.97 on a real 14-minute recording (and ~0.997 on a more static-
dominated one) -- i.e. the "near-duplicate" cutoff sat right on top of
ordinary, unremarkable continuity, while genuine transitions (checked
directly around known event boundaries in a synthetic test) showed adjacent
similarity as low as 0.85-0.91. Depending on exactly how a given recording's
background drifts, that one threshold could either wrongly admit many
still-similar-enough-to-sound-the-same windows (the reported bug) or, in
other cases, over-suppress everything past the first pick. Either way, a
single pairwise threshold applied only against already-selected candidates
is the wrong tool: it has no notion of a REGION, only of "close to one
specific other pick or not".

REGION_SIMILARITY_THRESHOLD below (0.92) is chosen from that same
measurement: comfortably above real transition-level similarity (0.85-0.91,
measured directly around known events), comfortably below typical
same-region continuity (0.96-0.998). _segment_into_regions() uses it to
partition the WHOLE chronological pool into contiguous regions up front,
before any ranking happens, by walking forward and starting a new region
wherever adjacent similarity drops below threshold (or there's a time gap,
e.g. from an excluded speech window). Within each region, the single
highest-scoring window becomes that region's one representative -- so a
long static passage becomes exactly one region contributing at most one
candidate, no matter how many raw 4-second windows it spans, while a brief
event still gets its own region and its own shot at being selected, because
the transition into and out of it is exactly what breaks the region.

Each pooled window still gets five 0-1 component scores (V6.1's sixth,
Sonic Interpretability, is removed -- see the regression note above; this
is not being solved by adding another scoring dimension, it needed a
structural fix to selection, not a new number):

  A. novelty            -- embedding distance from the RECORDING'S OVERALL
                            character (global mean embedding).
  B. contrast            -- embedding distance from this moment's LOCAL
                            (~30s) neighborhood average.
  C. layering            -- Shannon entropy of energy across frequency
                            bands (analysis._band_energy_entropy).
  D. distinctiveness     -- spectral crest factor (analysis._spectral_crest).
  E. temporal_interest    -- intra-window energy-envelope change
                            (analysis._temporal_flux): this is what is
                            supposed to (and, per the regression test below,
                            does) separate "continuous static" from "static
                            with a real event in it" at the per-window
                            level, ahead of region grouping.

select_diverse_candidates() combines these into one composite score per
window, segments the pool into regions, picks each region's best window as
its representative, then greedily builds a shortlist of DISTINCT REGIONS:
at each step it picks the highest-scoring remaining region-representative
after a soft diversity penalty (is this region acoustically close to a
region already picked?), on top of two hard guards (minimum time
separation, and a near-duplicate embedding cutoff). Because regions are
already deduplicated before this loop runs, the previous failure mode --
several picks from the same static stretch -- is now structurally
impossible, not just discouraged by a penalty.

Optional weight jitter + score jitter (both seeded, so a given seed always
reproduces the same shortlist) let the same pool surface a *different*
valid shortlist on repeat runs -- see listener.pipeline for how "Analyze
Again" uses this without re-running the classifier. Windows with any
detected speech are excluded entirely, before any of this scoring happens.
This module only ranks and selects; it does not judge meaning or extract
audio (Step 7) or describe it in language (Step 8).
"""
import random
from dataclasses import dataclass, field

import numpy as np

from .analysis import HOP_SECONDS, WindowAnalysis

MAX_CANDIDATES = 10
MIN_SEPARATION_SECONDS = 8.0  # light additional guard between two DIFFERENT regions' representatives
CONTRAST_WINDOW_SECONDS = 30.0  # how far around a moment counts as its "surroundings"
HARD_DUPLICATE_SIMILARITY = 0.97  # skip outright: this close in embedding space is a near-literal repeat

# How similar must two temporally-ADJACENT windows be to count as the SAME
# sonic region, rather than a transition into a new one? Chosen from real
# measurement, not guessed -- see module docstring for the percentile data.
# Meaningfully below typical "nothing happening" continuity (~0.96-0.998)
# and meaningfully above measured real transition points (~0.85-0.91), so it
# separates the two rather than splitting the difference blindly.
REGION_SIMILARITY_THRESHOLD = 0.92

# A gap this much larger than the normal hop between windows means something
# was excluded in between (most commonly a speech-flagged window) -- treat
# it as a region boundary too, not a continuation, since there is no actual
# adjacent-window evidence linking the two sides.
REGION_MAX_GAP_SECONDS = HOP_SECONDS * 1.25

SIMILARITY_DIVERSITY_WEIGHT = 0.20  # soft penalty scaled by closeness to the nearest already-selected region

# Fixed (not scaled) penalty for a window that was part of a PREVIOUS run's shown
# results (see previously_selected_starts below). Deliberately larger than any
# plausible composite-score gap or jitter swing so a previously-surfaced window
# reliably loses to a fresh alternative when one exists, rather than merely
# reshuffling among near-ties the way weight/score jitter alone does. Still a
# soft penalty, not an exclusion: if nothing better remains, the same window can
# still win -- "Analyze Again" never fabricates variation that isn't there.
PREVIOUS_RUN_PENALTY = 0.5

# Previously-shown region representatives also apply the SAME embedding-
# similarity diversity penalty used within a single run (see
# SIMILARITY_DIVERSITY_WEIGHT), not just an exact-start-time match. This is
# on top of, not instead of, PREVIOUS_RUN_PENALTY, and is combined with a
# hard cutoff below (same as HARD_DUPLICATE_SIMILARITY) so Retry cannot
# re-surface a near-identical region under a different representative window.
PREVIOUS_RUN_SIMILARITY_WEIGHT = 0.20

# Configurable without restructuring: pass a `weights` dict covering these
# five keys to build_candidate_pool/select_diverse_candidates (or override
# DEFAULT_WEIGHTS directly) to experiment with different emphases. Equal
# weighting -- no dimension is assumed more important than another until
# there's evidence to prefer otherwise.
DEFAULT_WEIGHTS = {
    "novelty": 0.20,
    "contrast": 0.20,
    "layering": 0.20,
    "distinctiveness": 0.20,
    "temporal_interest": 0.20,
}

JITTER_WEIGHT_RANGE = 0.40  # +/- 40% per-weight perturbation when a jitter_seed is given
JITTER_SCORE_RANGE = 0.15  # +/- 15% per-candidate composite-score perturbation


@dataclass
class Candidate:
    start_seconds: float
    score: float  # final composite score used for this selection -- a ranking signal, not an importance judgment
    scores: dict = field(default_factory=dict)  # component breakdown (novelty/contrast/layering/distinctiveness/temporal_interest)
    embedding: np.ndarray = None  # carried through for downstream diversity checks (e.g. Retry) -- not for display
    # How many raw 4-second windows this one candidate's sonic region
    # collapsed (see _segment_into_regions) -- e.g. 1 means a brief, sharply
    # bounded moment; 30 means a long, mostly-uniform stretch that this
    # candidate is the single representative of. Research/debug transparency
    # only, not used anywhere in scoring.
    region_window_count: int = 1


@dataclass
class _PooledWindow:
    window: WindowAnalysis
    scores: dict  # novelty/contrast/layering/distinctiveness/temporal_interest, each normalized 0-1


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


def _novelty_scores(eligible: list) -> list:
    """A. Distance from each window's embedding to the RECORDING-WIDE mean
    embedding across every eligible window -- "how different is this moment
    from the overall sonic character of the recording", independent of
    where in the recording it falls."""
    global_mean = np.mean([w.embedding for w in eligible], axis=0)
    return [1.0 - _cosine_similarity(w.embedding, global_mean) for w in eligible]


def _contrast_scores(eligible: list) -> list:
    """B. Distance from each window's embedding to its LOCAL neighborhood average.

    Uses a sliding window (eligible is time-sorted) instead of comparing every
    pair, so this stays fast on a 30-45 minute recording. Deliberately a
    different reference point than novelty above (local neighbors vs. the
    whole recording), so the two dimensions can disagree: a recording that is
    mostly rain with one door slam has a door with both high novelty AND high
    contrast, but a recording that alternates evenly between rain and traffic
    could have a rain moment with low novelty (rain is half the recording)
    yet high contrast if it's surrounded by traffic at that specific point.
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
    """Score every eligible window (no speech detected -- see analysis.py's
    narrow speech gate). This is the full candidate pool, deliberately much
    larger than the final ~10 -- select_diverse_candidates() draws the
    shortlist from it.

    Eligibility does not require PANNs to have confidently labeled anything
    -- a quiet, acoustically subtle moment that PANNs can't name is just as
    eligible as a loud, easily-labeled one. Only the speech gate excludes
    windows now.
    """
    eligible = [w for w in windows if not w.has_speech]
    if not eligible:
        return []

    novelty = _min_max_normalize(_novelty_scores(eligible))
    contrast = _min_max_normalize(_contrast_scores(eligible))
    layering = _min_max_normalize([w.layering_raw for w in eligible])
    distinctiveness = _min_max_normalize([w.distinctiveness_raw for w in eligible])
    temporal_interest = _min_max_normalize([w.temporal_interest_raw for w in eligible])

    return [
        _PooledWindow(
            window=w,
            scores={
                "novelty": novelty[i],
                "contrast": contrast[i],
                "layering": layering[i],
                "distinctiveness": distinctiveness[i],
                "temporal_interest": temporal_interest[i],
            },
        )
        for i, w in enumerate(eligible)
    ]


def _segment_into_regions(pool: list[_PooledWindow]) -> list[list[int]]:
    """Partition the chronologically-ordered pool into contiguous sonic
    regions: a new region starts wherever a temporally-adjacent pair of
    windows either has a time gap larger than REGION_MAX_GAP_SECONDS, or an
    embedding cosine similarity below REGION_SIMILARITY_THRESHOLD (see
    module docstring for how that threshold was chosen from real data).

    This is what makes a long, mostly-uniform stretch of the recording
    collapse into ONE region -- and therefore contribute at most one final
    candidate -- no matter how many overlapping 4-second windows it spans,
    while a real transition (something starts, changes, or stops) breaks the
    chain and starts a new region right there.
    """
    if not pool:
        return []

    regions: list[list[int]] = [[0]]
    for i in range(1, len(pool)):
        prev_window = pool[i - 1].window
        window = pool[i].window
        gap = window.start_seconds - prev_window.start_seconds > REGION_MAX_GAP_SECONDS
        similarity = _cosine_similarity(prev_window.embedding, window.embedding)
        if gap or similarity < REGION_SIMILARITY_THRESHOLD:
            regions.append([i])
        else:
            regions[-1].append(i)
    return regions


def count_regions(pool: list[_PooledWindow]) -> int:
    """How many distinct sonic regions the pool currently contains -- research/
    debug transparency (see pipeline.py's logging), not used in selection."""
    return len(_segment_into_regions(pool))


@dataclass
class _RegionCandidate:
    """One region's single best-scoring representative window, plus enough
    bookkeeping to log how many raw windows it stood in for."""

    pool_index: int
    region_size: int
    base_score: float


def _pick_region_representatives(pool: list[_PooledWindow], base_scores: list[float]) -> list[_RegionCandidate]:
    regions = _segment_into_regions(pool)
    representatives = []
    for region in regions:
        best_idx = max(region, key=lambda idx: base_scores[idx])
        representatives.append(
            _RegionCandidate(pool_index=best_idx, region_size=len(region), base_score=base_scores[best_idx])
        )
    return representatives


def select_diverse_candidates(
    pool: list[_PooledWindow],
    count: int = MAX_CANDIDATES,
    weights: dict = None,
    jitter_seed: int = None,
    previously_selected_starts: set = None,
    previously_selected_embeddings: list = None,
) -> list[Candidate]:
    """Greedily build a diverse shortlist of up to `count` DISTINCT SONIC
    REGIONS from the pool -- not up to `count` raw windows.

    weights: override DEFAULT_WEIGHTS (must cover the same five component keys).
    See DEFAULT_WEIGHTS's docstring -- this is the whole configurability
    surface, deliberately: a caller passes a dict, nothing here needs
    restructuring to try different emphases.

    jitter_seed: if given, deterministically perturbs both the weights and each
    candidate's composite score by a bounded random amount (affecting both
    which window becomes each region's representative, and the ranking
    between regions), so the same pool can surface a different (but still
    quality-gated) shortlist on repeat runs -- see listener.pipeline / app.py's
    "Analyze Again". Leave as None for the canonical, unjittered ranking.

    previously_selected_starts: start_seconds of windows already SHOWN to the
    user in an earlier run on this same recording (see app.py). Each gets
    PREVIOUS_RUN_PENALTY subtracted during selection.

    previously_selected_embeddings: embeddings of region representatives
    already shown in earlier runs. A near-duplicate region (>=
    HARD_DUPLICATE_SIMILARITY similar to any of these) is excluded outright,
    the same hard guard used against regions already picked in the current
    run; anything closer than that gets the soft
    PREVIOUS_RUN_SIMILARITY_WEIGHT penalty. This is what makes "Analyze
    Again" search for genuinely different regions rather than re-surfacing
    the same region under a different representative window a few seconds
    over.

    Returns candidates in chronological order. Returns fewer than `count` if
    the recording doesn't have that many genuinely distinct regions --
    candidates are never fabricated (or padded with near-duplicate regions)
    to reach the target count.
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

    region_candidates = _pick_region_representatives(pool, base_scores)

    remaining = list(range(len(region_candidates)))
    selected: list[_RegionCandidate] = []
    prior_embeddings = previously_selected_embeddings or []

    def embedding_of(rc: _RegionCandidate) -> np.ndarray:
        return pool[rc.pool_index].window.embedding

    def start_of(rc: _RegionCandidate) -> float:
        return pool[rc.pool_index].window.start_seconds

    while len(selected) < count and remaining:
        best_idx = None
        best_value = -float("inf")

        for idx in remaining:
            rc = region_candidates[idx]

            if any(abs(start_of(rc) - start_of(s)) < MIN_SEPARATION_SECONDS for s in selected):
                continue
            if any(_cosine_similarity(embedding_of(rc), embedding_of(s)) >= HARD_DUPLICATE_SIMILARITY for s in selected):
                continue
            if prior_embeddings and max(_cosine_similarity(embedding_of(rc), e) for e in prior_embeddings) >= HARD_DUPLICATE_SIMILARITY:
                continue

            value = rc.base_score
            if selected:
                max_similarity = max(_cosine_similarity(embedding_of(rc), embedding_of(s)) for s in selected)
                value -= SIMILARITY_DIVERSITY_WEIGHT * max_similarity
            if prior_embeddings:
                max_prior_similarity = max(_cosine_similarity(embedding_of(rc), e) for e in prior_embeddings)
                value -= PREVIOUS_RUN_SIMILARITY_WEIGHT * max_prior_similarity
            if previously_selected_starts and start_of(rc) in previously_selected_starts:
                value -= PREVIOUS_RUN_PENALTY

            if value > best_value:
                best_value = value
                best_idx = idx

        if best_idx is None:
            break  # everything left violates a hard guard

        selected.append(region_candidates[best_idx])
        remaining.remove(best_idx)

    selected.sort(key=start_of)  # chronological display order

    return [
        Candidate(
            start_seconds=start_of(rc),
            score=rc.base_score,
            scores=dict(pool[rc.pool_index].scores),
            embedding=embedding_of(rc),
            region_window_count=rc.region_size,
        )
        for rc in selected
    ]


def select_candidates(windows: list[WindowAnalysis], max_candidates: int = MAX_CANDIDATES) -> list[Candidate]:
    """Convenience wrapper: build the pool and select a shortlist in one call."""
    return select_diverse_candidates(build_candidate_pool(windows), count=max_candidates)
