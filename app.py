"""Listener -- Version 5.

Version 1 transforms an existing environmental recording into ~10 candidate
sonic moments (AI surfaces). Version 2 lets the person decide which ones to
keep (human selects). Version 3 lets them edit the ones they kept -- trim,
reorder, optionally crossfade, or remove -- into a composition of their own
making (human edits). Version 4 refined transitions/downloads. Version 5 lets
them explore a candidate through a chosen affective direction (Calm,
Nostalgic, ...) -- the system transforms the SAME clip, it does not decide
what the clip already means.

AI surfaces. Humans interpret. Humans edit. The system never labels a sound
as meaningful, important, or emotional -- only what was acoustically
detected. The AI's original candidate list is never reordered, filtered, or
re-ranked based on what the person keeps or how they edit it, so the three
can be compared: what the AI noticed, what the person chose, and how they
transformed it. There is no automatic sequencing or soundscape generation
anywhere in this file -- every ordering, trim, and crossfade decision below
is the user's own, made through an explicit control. The Version 5 affective
transform follows the same rule: the DIRECTION is always the user's own
choice from an explicit list, never an inference the system makes about the
sound -- see listener/affect.py.
"""
import shutil
import uuid
from pathlib import Path

import streamlit as st

from listener.affect import TONES, TRANSFORM_DIR, transform_clip
from listener.analysis import WINDOW_SECONDS, analyze_recording
from listener.audio_io import SUPPORTED_EXTENSIONS, format_timestamp, load_recording
from listener.clipping import CLIP_DIR
from listener.composition import COMPOSITION_DIR, CompositionItem, build_composition
from listener.description import describe_candidate
from listener.pipeline import generate_verified_candidates
from listener.research_log import log_composition_snapshot, log_decision

st.set_page_config(page_title="Listener", page_icon="🎧")

st.title("LISTENER")
st.caption("Surface moments from an everyday sound recording.")

st.subheader("Upload one audio recording")
uploaded_file = st.file_uploader(
    "Upload one audio recording",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    if st.session_state.get("_uploaded_name") != uploaded_file.name:
        with st.spinner("Reading recording..."):
            st.session_state["recording"] = load_recording(uploaded_file)
        st.session_state["_uploaded_name"] = uploaded_file.name
        # a new upload invalidates any previous analysis, decisions, and edits
        for key in (
            "results", "decisions", "run_id", "composition", "removed", "show_editor",
            "windows", "analysis_number", "shown_window_starts", "affect_state",
        ):
            st.session_state.pop(key, None)

    recording = st.session_state["recording"]

    st.write(f"**Filename:** {recording.filename}")
    st.write(f"**Duration:** {format_timestamp(recording.duration_seconds)}")
    st.audio(recording.filepath)

    def _select_and_extract(analysis_number: int, jitter_seed) -> None:
        """(Re)select and extract candidates from the already-analyzed windows,
        without re-running the classifier over the whole recording. Used by
        both "Analyze Recording" (analysis_number=1, no jitter -- the
        canonical ranking) and "Analyze Again" (a new seed each time, so the
        same pool of candidate windows surfaces a different valid shortlist)."""
        if CLIP_DIR.exists():
            shutil.rmtree(CLIP_DIR)
        if TRANSFORM_DIR.exists():
            shutil.rmtree(TRANSFORM_DIR)
        st.session_state["affect_state"] = {}
        windows = st.session_state["windows"]
        previously_selected_starts = st.session_state.get("shown_window_starts", set())
        candidates, clips = generate_verified_candidates(
            recording.filepath, recording.duration_seconds, windows, WINDOW_SECONDS,
            jitter_seed=jitter_seed, run_id=st.session_state["run_id"], analysis_number=analysis_number,
            previously_selected_starts=previously_selected_starts,
        )
        descriptions = [describe_candidate(c) for c in candidates]
        # accumulate across runs (not replace) so "Analyze Again" keeps steering
        # away from everything shown so far this session, not just the last run
        st.session_state["shown_window_starts"] = previously_selected_starts | {c.start_seconds for c in candidates}

        st.session_state["results"] = list(zip(clips, descriptions))
        st.session_state["decisions"] = {}  # candidate_id -> "kept" | "discarded"
        st.session_state["decision_sequence"] = 0
        st.session_state["analysis_number"] = analysis_number
        # a new candidate set starts a fresh edit too -- old composition referenced clips
        # that may no longer exist, and candidate numbers may now mean different sounds
        st.session_state.pop("composition", None)
        st.session_state.pop("removed", None)
        st.session_state["show_editor"] = False

    if st.button("Analyze Recording"):
        st.session_state["run_id"] = uuid.uuid4().hex[:8]

        progress_bar = st.progress(0.0, text="Analyzing your recording...")

        def _update_progress(fraction: float) -> None:
            progress_bar.progress(
                fraction, text=f"Finding distinct environmental sound moments... ({fraction * 100:.0f}%)"
            )

        st.session_state["windows"] = analyze_recording(recording.filepath, progress_callback=_update_progress)
        _select_and_extract(analysis_number=1, jitter_seed=None)
        progress_bar.empty()

    results = st.session_state.get("results")
    if results is not None:
        run_id = st.session_state["run_id"]
        decisions = st.session_state["decisions"]
        analysis_number = st.session_state.get("analysis_number", 1)

        st.divider()

        if not results:
            st.write("No distinct candidate sound moments were detected in this recording.")
        elif len(results) < 10:
            st.write(f"Fewer than 10 distinct candidate moments were detected: **{len(results)}** found.")
        else:
            st.write(f"**{len(results)} candidate sound moments found.**")

        if analysis_number > 1:
            st.caption(f"Analysis #{analysis_number} — a new candidate set from the same recording.")

        if st.button("Analyze Again"):
            with st.spinner("Exploring a different set of candidate moments from the same recording..."):
                _select_and_extract(analysis_number=analysis_number + 1, jitter_seed=analysis_number)
            st.rerun()

        for i, (clip, description) in enumerate(results, start=1):
            st.divider()
            st.markdown(f"**Candidate {i:02d}**")
            st.audio(clip.filepath)
            st.write(description.text)
            if description.detected_sounds:
                st.write("Detected sounds:")
                for sound in description.detected_sounds:
                    label = sound.display_label
                    st.write(f"- {label[0].upper()}{label[1:]}")
            duration = clip.end_seconds - clip.start_seconds
            st.write(
                f"{format_timestamp(clip.start_seconds)} – {format_timestamp(clip.end_seconds)} "
                f"({duration:.1f}s)"
            )
            if description.detected_sounds:
                with st.expander("Debug: raw classifier output"):
                    st.caption("Research transparency -- what PANNs actually returned for this exact clip.")
                    for sound in description.detected_sounds:
                        st.write(f"- `{sound.raw_label}` -- confidence {sound.confidence:.2f} -> \"{sound.display_label}\"")

            keep_col, discard_col, download_col = st.columns(3)
            if keep_col.button("Keep", key=f"keep_{i}"):
                decisions[i] = "kept"
                st.session_state["decision_sequence"] += 1
                log_decision(run_id, i, "kept", st.session_state["decision_sequence"])
            if discard_col.button("Discard", key=f"discard_{i}"):
                decisions[i] = "discarded"
                st.session_state["decision_sequence"] += 1
                log_decision(run_id, i, "discarded", st.session_state["decision_sequence"])
            # Downloads the actual candidate .wav the pipeline already exported --
            # not regenerated or re-classified, and unaffected by Keep/Discard,
            # since neither decision deletes or modifies the clip file on disk.
            # round (not truncate) to match format_timestamp's display above, so
            # the filename's timestamp always agrees with what's shown on screen
            minutes, seconds = divmod(int(round(clip.start_seconds)), 60)
            download_col.download_button(
                "Download clip",
                data=Path(clip.filepath).read_bytes(),
                file_name=f"candidate_{i:02d}_{minutes:02d}m{seconds:02d}s.wav",
                mime="audio/wav",
                key=f"download_{i}",
            )

            state = decisions.get(i)
            if state == "kept":
                st.write("✓ Kept")
            elif state == "discarded":
                st.write("Discarded")

            # ------------------------------------------------------------
            # Version 5: explore an affective direction for this candidate.
            # The direction is always the user's own choice from an explicit
            # list -- the system transforms the clip according to it, it
            # never decides what the clip already means. See listener/affect.py.
            # ------------------------------------------------------------
            with st.expander("Explore an emotional interpretation"):
                st.caption(
                    "Choose a direction below. The system transforms this exact "
                    "recording through that direction -- it does not decide what "
                    "the sound itself means."
                )
                tones_by_label = {t.label: t for t in TONES}
                chosen_label = st.selectbox(
                    "Affective direction", list(tones_by_label), key=f"affect_tone_{i}",
                    format_func=lambda label: f"{label} ({tones_by_label[label].descriptors})",
                )
                chosen_tone = tones_by_label[chosen_label]

                affect_state = st.session_state["affect_state"]
                state_key = str(i)
                current = affect_state.get(state_key)

                gen_col, again_col = st.columns(2)
                if gen_col.button("Generate transformation", key=f"affect_gen_{i}"):
                    path = transform_clip(clip.filepath, chosen_tone.key, 1)
                    affect_state[state_key] = {"tone_key": chosen_tone.key, "version": 1, "path": path}
                    st.rerun()

                current = affect_state.get(state_key)
                if current and current["tone_key"] == chosen_tone.key:
                    st.write("Original sound")
                    st.audio(clip.filepath)
                    st.write(f"{chosen_tone.label} interpretation -- Version {current['version']}")
                    st.audio(current["path"])
                    if again_col.button("Try another interpretation", key=f"affect_again_{i}"):
                        next_version = current["version"] + 1
                        path = transform_clip(clip.filepath, chosen_tone.key, next_version)
                        affect_state[state_key] = {
                            "tone_key": chosen_tone.key, "version": next_version, "path": path,
                        }
                        st.rerun()

        st.divider()
        kept_ids = [i for i in range(1, len(results) + 1) if decisions.get(i) == "kept"]
        st.markdown("### My Selected Sounds")
        st.write(f"You selected {len(kept_ids)} sounds.")

        for i, (clip, description) in enumerate(results, start=1):
            if i not in kept_ids:
                continue
            st.markdown(f"**Candidate {i:02d}**")
            st.audio(clip.filepath)
            st.write(description.text)
            st.write(f"{format_timestamp(clip.start_seconds)} – {format_timestamp(clip.end_seconds)}")

        # ------------------------------------------------------------------
        # Version 3: editing workspace
        # ------------------------------------------------------------------
        if kept_ids:
            if st.button("Edit Selected Sounds"):
                if "composition" not in st.session_state:
                    composition = []
                    for i in kept_ids:
                        clip, _description = results[i - 1]
                        clip_duration = clip.end_seconds - clip.start_seconds
                        composition.append(
                            CompositionItem(
                                candidate_id=i,
                                clip_filepath=clip.filepath,
                                clip_start_seconds=clip.start_seconds,
                                clip_end_seconds=clip.end_seconds,
                                trim_start=0.0,
                                trim_end=clip_duration,
                                crossfade_into_next=False,
                            )
                        )
                    st.session_state["composition"] = composition
                    st.session_state["removed"] = []
                    st.session_state["composition_needs_rebuild"] = True
                st.session_state["show_editor"] = True

        if st.session_state.get("show_editor") and st.session_state.get("composition") is not None:
            composition = st.session_state["composition"]
            removed = st.session_state["removed"]
            descriptions_by_id = {i: results[i - 1][1] for i in range(1, len(results) + 1)}

            def _record_composition_change() -> None:
                # every trim/reorder/remove/restore/crossfade edit calls this, so the
                # (possibly now-wrong) old preview never keeps playing silently below
                st.session_state["composition_needs_rebuild"] = True
                log_composition_snapshot(run_id, composition, removed)

            st.divider()
            st.header("Editing Workspace")
            st.caption(
                "Trim, reorder, remove, or crossfade the sounds you kept. "
                "This does not change what the AI originally surfaced."
            )

            if not composition:
                st.write("No sounds remain in this composition. Restore one below, or select more sounds above.")

            for pos, item in enumerate(composition):
                st.divider()
                clip_duration = item.clip_end_seconds - item.clip_start_seconds
                st.markdown(f"**Sound {item.candidate_id}** (position {pos + 1} of {len(composition)})")
                st.write(descriptions_by_id[item.candidate_id].text)

                # A. play (native play/pause/replay controls), reflecting the current trim
                st.audio(item.clip_filepath, start_time=item.trim_start, end_time=item.trim_end)

                # B. trim
                trim_range = st.slider(
                    "Trim (seconds within this clip)",
                    min_value=0.0,
                    max_value=clip_duration,
                    value=(item.trim_start, item.trim_end),
                    step=0.1,
                    key=f"trim_{item.candidate_id}",
                )
                new_start, new_end = trim_range
                if new_end - new_start < 0.2:
                    new_end = min(clip_duration, new_start + 0.2)
                if (new_start, new_end) != (item.trim_start, item.trim_end):
                    item.trim_start, item.trim_end = new_start, new_end
                    _record_composition_change()

                # C. reorder
                move_up_col, move_down_col, remove_col = st.columns(3)
                if move_up_col.button("Move Up", key=f"up_{item.candidate_id}", disabled=(pos == 0)):
                    composition[pos - 1], composition[pos] = composition[pos], composition[pos - 1]
                    _record_composition_change()
                    st.rerun()
                if move_down_col.button(
                    "Move Down", key=f"down_{item.candidate_id}", disabled=(pos == len(composition) - 1)
                ):
                    composition[pos + 1], composition[pos] = composition[pos], composition[pos + 1]
                    _record_composition_change()
                    st.rerun()

                # E. remove (does not touch the original V2 kept list)
                if remove_col.button("Remove from composition", key=f"remove_{item.candidate_id}"):
                    composition.pop(pos)
                    removed.append(item)
                    _record_composition_change()
                    st.rerun()

                # D. optional simple crossfade into the next clip
                if pos < len(composition) - 1:
                    crossfade = st.checkbox(
                        "Crossfade into next clip",
                        value=item.crossfade_into_next,
                        key=f"crossfade_{item.candidate_id}",
                    )
                    if crossfade != item.crossfade_into_next:
                        item.crossfade_into_next = crossfade
                        _record_composition_change()

            if removed:
                st.divider()
                st.markdown("**Removed from composition**")
                st.caption("These stay part of your selected sounds -- restore any of them here.")
                for item in list(removed):
                    label_col, restore_col = st.columns([3, 1])
                    label_col.write(f"Sound {item.candidate_id} -- {descriptions_by_id[item.candidate_id].text}")
                    if restore_col.button("Restore", key=f"restore_{item.candidate_id}"):
                        removed.remove(item)
                        composition.append(item)
                        _record_composition_change()
                        st.rerun()

            # ------------------------------------------------------------------
            # Composition preview
            # ------------------------------------------------------------------
            st.divider()
            st.markdown("### My Sound Composition")
            if composition:
                st.write(" → ".join(f"[Sound {item.candidate_id}]" for item in composition))
            else:
                st.write("(empty)")

            needs_rebuild = st.session_state.get("composition_needs_rebuild", True)

            if composition:
                preview_clicked = st.button(
                    "▶ Preview Composition",
                    type="primary" if needs_rebuild else "secondary",
                )
                if preview_clicked:
                    if COMPOSITION_DIR.exists():
                        shutil.rmtree(COMPOSITION_DIR)
                    with st.spinner("Building your composition..."):
                        output_path = build_composition(composition, run_id)
                    log_composition_snapshot(run_id, composition, removed)
                    st.session_state["composition_output"] = output_path
                    st.session_state["composition_needs_rebuild"] = False
                    # the button above already rendered primary/red for this pass (it was
                    # drawn before this click was processed) -- rerun so it immediately
                    # reflects "up to date" instead of looking stale for one extra render
                    st.rerun()

            # ------------------------------------------------------------------
            # Final output
            # ------------------------------------------------------------------
            composition_output = st.session_state.get("composition_output")
            if composition or composition_output:
                st.divider()
                st.markdown("### Preview Soundscape")

                if needs_rebuild:
                    # Never play a stale build: the player itself doesn't render
                    # until the user explicitly rebuilds after their latest edit.
                    if composition_output:
                        st.info(
                            "Your composition has changed since this was built. "
                            "Click ▶ Preview Composition above to hear your latest edits."
                        )
                    else:
                        st.write("Click ▶ Preview Composition above to generate your soundscape.")
                else:
                    st.audio(composition_output)
                    with open(composition_output, "rb") as f:
                        st.download_button(
                            "Download composition",
                            data=f.read(),
                            file_name="listener_composition.wav",
                            mime="audio/wav",
                        )
