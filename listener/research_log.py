"""Lightweight research logging: how the AI selected candidates (Step 6), and
V2 Keep/Discard decisions and V3 editing.

Records events to local, append-only JSONL files so the AI's original
candidate list can later be compared against what the person actually chose
to keep, and against how they transformed it -- and so the selection process
itself (why these 10 moments and not others) is inspectable, not a black
box. Intentionally minimal -- no dashboard, no aggregation, just a log of
what happened.
"""
import json
import time
from pathlib import Path

ANALYSIS_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "logs" / "analysis.jsonl"
DECISIONS_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "logs" / "decisions.jsonl"
COMPOSITION_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "logs" / "composition.jsonl"


def log_analysis_run(
    run_id: str,
    analysis_number: int,
    jitter_seed,
    classifier_name: str,
    total_windows: int,
    speech_excluded_windows: int,
    pool_size: int,
    region_count: int,
    shortlist_size: int,
    speech_clip_excluded: int,
    final_candidates: list,
) -> None:
    """Append one record of how a single "Analyze Recording"/"Analyze Again"
    run arrived at its final candidates: model used, how many windows/
    candidates were considered and why some were excluded, and the full
    component-score breakdown behind each final pick. Research transparency
    -- never shown in the main UI. Never raises; logging shouldn't break the app.

    region_count (V6.2): how many distinct sonic regions build_candidate_pool's
    output was segmented into BEFORE diversity selection ran (see
    selection._segment_into_regions) -- i.e. the true number of acoustically
    distinct moments available. Compare against each candidate's
    region_window_count below (how many raw windows that one pick's region
    collapsed) to see how much redundant near-duplicate content one candidate
    is standing in for.
    """
    try:
        ANALYSIS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "run_id": run_id,
            "analysis_number": analysis_number,
            "jitter_seed": jitter_seed,
            "at": time.time(),
            "classifier": classifier_name,
            "total_windows_analyzed": total_windows,
            "excluded_speech_windows": speech_excluded_windows,
            "pool_size": pool_size,
            "region_count": region_count,
            "shortlist_size": shortlist_size,
            "excluded_speech_in_clip": speech_clip_excluded,
            "final_count": len(final_candidates),
            "final_candidates": [
                {
                    "start_seconds": c.start_seconds,
                    "region_window_count": c.region_window_count,
                    "score": c.score,
                    "component_scores": c.scores,
                }
                for c in final_candidates
            ],
        }
        with open(ANALYSIS_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def log_decision(run_id: str, candidate_id: int, decision: str, sequence: int) -> None:
    """Append one V2 Keep/Discard event. Never raises -- logging shouldn't break the app."""
    try:
        DECISIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "decision": decision,
            "decided_at": time.time(),
            "sequence": sequence,
        }
        with open(DECISIONS_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def log_composition_snapshot(run_id: str, items: list, removed_items: list) -> None:
    """Append one snapshot of the V3 composition state: order, trims, crossfades, removals.

    `items` and `removed_items` are both lists of CompositionItem -- the workspace keeps
    full items (not just ids) for removed sounds so the "Restore" UI can show what they were.
    """
    try:
        COMPOSITION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "run_id": run_id,
            "at": time.time(),
            "final_order": [item.candidate_id for item in items],
            "removed": [item.candidate_id for item in removed_items],
            "items": [
                {
                    "candidate_id": item.candidate_id,
                    "original_start_seconds": item.clip_start_seconds,
                    "original_end_seconds": item.clip_end_seconds,
                    "trim_start": item.trim_start,
                    "trim_end": item.trim_end,
                    "edited_duration": item.trim_end - item.trim_start,
                    "crossfade_into_next": item.crossfade_into_next,
                }
                for item in items
            ],
        }
        with open(COMPOSITION_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
