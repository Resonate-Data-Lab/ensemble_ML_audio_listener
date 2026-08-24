"""Ties Step 6 (select) and Step 7 (extract) together with a speech safety-check
on the actual exported audio, and logs how each run arrived at its candidates.

select_diverse_candidates() already excludes windows with any detected speech
label (not just a dominant one), using a ~4s analysis window. But the ~5-6.5s
clip extracted around a selected window repositions its boundary using pure
audio energy (see clipping._clip_bounds), which can drift a couple of seconds
onto adjacent speech that was never classified. So a candidate that looked
speech-free at selection time can still end up with audible speech in its
final clip.

generate_verified_candidates() re-checks each candidate's actual exported clip
in short sub-windows (see analysis.clip_contains_speech) and drops anything
that contains identifiable speech anywhere in it, backfilling from a larger
shortlist so the final count still aims for ~10 when the recording has enough
genuinely distinct, speech-free moments. Nothing here judges meaning or ranks
by importance -- that's still entirely selection.py's job; this module only
adds one more objective check (is there identifiable speech in this exact
audio) before a candidate is allowed into the final list, and records what
happened along the way.

weights/jitter_seed pass straight through to select_diverse_candidates, so
"Analyze Again" (app.py) can request a different valid shortlist from the
SAME already-computed windows, without re-running the classifier.
"""
from pathlib import Path

from .analysis import clip_contains_speech
from .audio_classifier import get_classifier
from .clipping import Clip, extract_clips
from .research_log import log_analysis_run
from .selection import (
    Candidate, MAX_CANDIDATES, build_candidate_pool, count_regions, select_diverse_candidates,
)

SHORTLIST_SIZE = 25  # generous shortlist to draw backfill candidates from


def generate_verified_candidates(
    recording_filepath: str,
    recording_duration: float,
    windows: list,
    window_seconds: float,
    weights: dict = None,
    jitter_seed: int = None,
    run_id: str = None,
    analysis_number: int = 1,
    previously_selected_starts: set = None,
    previously_selected_embeddings: list = None,
) -> tuple:
    """Return up to MAX_CANDIDATES (candidate, clip) pairs whose exported clips
    contain no identifiable speech, in chronological order.

    Speech-contaminated candidates are dropped entirely (their clip file is
    deleted) and replaced with the next-best candidate from a larger
    shortlist. If the recording doesn't have enough non-speech alternatives,
    fewer than MAX_CANDIDATES are returned rather than fabricated.

    previously_selected_starts / previously_selected_embeddings: passed
    straight through to select_diverse_candidates (see there) so "Analyze
    Again" steers away from windows already shown in an earlier run on this
    recording -- both by exact start-time match and by acoustic similarity.
    """
    pool = build_candidate_pool(windows)
    speech_excluded_windows = sum(1 for w in windows if w.has_speech)
    region_count = count_regions(pool)  # how many distinct sonic regions existed BEFORE selection

    shortlist = select_diverse_candidates(
        pool, count=SHORTLIST_SIZE, weights=weights, jitter_seed=jitter_seed,
        previously_selected_starts=previously_selected_starts,
        previously_selected_embeddings=previously_selected_embeddings,
    )
    if not shortlist:
        if run_id:
            log_analysis_run(
                run_id, analysis_number, jitter_seed, type(get_classifier()).__name__,
                len(windows), speech_excluded_windows, len(pool), region_count, 0, 0, [],
            )
        return [], []

    shortlist_clips = extract_clips(recording_filepath, recording_duration, shortlist, window_seconds)

    candidates: list[Candidate] = []
    clips: list[Clip] = []
    speech_clip_excluded = 0
    for candidate, clip in zip(shortlist, shortlist_clips):
        if len(candidates) >= MAX_CANDIDATES:
            Path(clip.filepath).unlink(missing_ok=True)
            continue
        if clip_contains_speech(clip.filepath):
            speech_clip_excluded += 1
            Path(clip.filepath).unlink(missing_ok=True)
            continue
        candidates.append(candidate)
        clips.append(clip)

    if run_id:
        log_analysis_run(
            run_id=run_id,
            analysis_number=analysis_number,
            jitter_seed=jitter_seed,
            classifier_name=type(get_classifier()).__name__,
            total_windows=len(windows),
            speech_excluded_windows=speech_excluded_windows,
            pool_size=len(pool),
            region_count=region_count,
            shortlist_size=len(shortlist),
            speech_clip_excluded=speech_clip_excluded,
            final_candidates=candidates,
        )

    return candidates, clips
