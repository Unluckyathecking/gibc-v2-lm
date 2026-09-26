# Scene 1 voice samples (OpenRouter TTS)

Each clip is the ~30 s opening scene, rendered with `--sample --tts openrouter`; the `.mp3` is the raw model output, the `.mp4` the slide plus that audio loudness-normalised. Style prompt, where the model takes one: "Read in a calm, measured British documentary narrator voice, unhurried, clear diction." Re-render all six with `zsh video/voice_samples/render.sh`.

| Clip | Model | Voice | Style prompt | Seconds | Cost |
|---|---|---|---|---|---|
| gemini-3.8-flash-tts_charon | google/gemini-3.8-flash-tts | Charon | yes | not rendered: HTTP 402, no OpenRouter credit | - |
| gemini-3.8-flash-tts_iapetus | google/gemini-3.8-flash-tts | Iapetus | yes | not rendered: HTTP 402 | - |
| gemini-3.8-flash-tts_kore | google/gemini-3.8-flash-tts | Kore | yes | not rendered: HTTP 402 | - |
| qwen-audio-3.0-tts-plus_longanlingxin | qwen/qwen-audio-3.0-tts-plus | longanlingxin (female, "warm and empathetic"; the only other voice, longanlufeng, is "bright and cheerful") | not supported | not rendered: HTTP 402 | - |
| fish-s2.1-pro-free_david-british-documentary | fish-audio/s2.1-pro-free:free | daf86fe840734f1cac435434e562ea2a ("david - british documentary" on fish.audio) | not supported | 29.9 audio, 30.8 clip | $0 (free) |
| minimax-speech-2.8-hd_expressive-narrator | minimax/speech-2.8-hd | English_expressive_narrator | not supported | not rendered: HTTP 402 | - |
