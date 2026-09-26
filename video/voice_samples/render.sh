#!/bin/zsh
# Re-render the scene-1 voice comparison clips via OpenRouter. Needs OPENROUTER_API_KEY in the
# environment and credit on the account. Run from anywhere: zsh video/voice_samples/render.sh
cd "${0:A:h}/../.."
STYLE="Read in a calm, measured British documentary narrator voice, unhurried, clear diction."
run() { slug=$1; shift; echo "== $slug"
  uv run python scripts/make_video.py --ckpt runs/final.pt --tok runs/tokenizer.json --sample \
    --tts openrouter "$@" --out video/voice_samples/$slug.mp4 | grep -E "wrote|cost"; }
run gemini-3.8-flash-tts_charon  --model google/gemini-3.8-flash-tts --voice Charon  --style "$STYLE"
run gemini-3.8-flash-tts_iapetus --model google/gemini-3.8-flash-tts --voice Iapetus --style "$STYLE"
run gemini-3.8-flash-tts_kore    --model google/gemini-3.8-flash-tts --voice Kore    --style "$STYLE"
run qwen-audio-3.0-tts-plus_longanlingxin --model qwen/qwen-audio-3.0-tts-plus --voice longanlingxin
run fish-s2.1-pro-free_david-british-documentary --model fish-audio/s2.1-pro-free:free \
    --voice daf86fe840734f1cac435434e562ea2a
run minimax-speech-2.8-hd_expressive-narrator --model minimax/speech-2.8-hd --voice English_expressive_narrator
rmdir video/voice_samples/slides 2>/dev/null
