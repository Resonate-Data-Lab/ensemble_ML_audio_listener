"""Listener: transforms an existing environmental recording into candidate sonic moments,
lets a person decide which ones to keep, and lets them edit the ones they kept.

Modules are split by pipeline stage so later versions can be added without
rewriting earlier stages:

    audio_io      -- V1 Step 1-2: load an uploaded recording, read duration
    segmentation  -- V1 Step 3:   split long recordings into analyzable chunks
    analysis      -- V1 Step 4-5: detect sound events, changes, speech
    selection     -- V1 Step 6:   rank and choose diverse candidate moments
    clipping      -- V1 Step 7:   extract ~5s clips from the original recording
    description   -- V1 Step 8:   neutral descriptions + sound labels
    composition   -- V3:          trim/reorder/crossfade the user's kept clips into one file
    research_log  -- V2/V3:       append-only logs of Keep/Discard and editing decisions
"""
