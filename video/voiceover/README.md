# Your voiceover takes

`scripts/teleprompter.py` saves one take per scene here: `scene_01.webm` ... `scene_08.webm` (or `.m4a`/`.wav`/`.ogg`, depending on the browser), each with a `scene_NN.json` giving its length after silence trimming. A retake overwrites the scene's file. You can also drop in takes recorded elsewhere, named `scene_NN.(webm|wav|m4a|mp3|ogg)`.

Check them with `uv run python scripts/make_video.py --voiceover-check`, render with `--tts voiceover`. See "Recording your own voiceover" in `docs/VIDEO_SCRIPT.md`.
