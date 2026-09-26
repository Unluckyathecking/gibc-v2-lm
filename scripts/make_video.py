"""Render the narrated demo video from video/script.json, end to end, on a Mac.

Usage:
  uv run python scripts/make_video.py --ckpt runs/final.pt --tok runs/tokenizer.json
  uv run python scripts/make_video.py --ckpt ... --tok ... --script video/script.json \\
      --out video/gibc_demo.mp4 [--voice bm_george] [--tts say --voice Daniel] [--reuse-demo]
  uv run python scripts/make_video.py --ckpt ... --tok ... --sample --voice af_heart
  uv run python scripts/make_video.py --ckpt ... --tok ... --tts openrouter \\
      --model google/gemini-3.8-flash-tts --voice Charon --style "Read in a calm ..."

Pipeline: substitute {{KEY}} values into script.json; synthesise each narration
sentence with Kokoro-82M (local neural TTS, default) or macOS `say`, or each scene
with an OpenRouter TTS model (needs OPENROUTER_API_KEY; MP3 decoded by ffmpeg); render one
1920x1080 slide per scene with PIL; run scripts/demo.py's sampler on the demo prompts
and animate the output as a typewriter; stream every frame into one ffmpeg process with
the concatenated narration, loudness-normalised to -16 LUFS.
Writes the mp4, video/captions.srt, video/slides/*.png and video/demo_output.json.
--sample renders only the first scene to video/voice_sample.mp4, for judging a voice.
Needs ffmpeg/ffprobe on PATH; Kokoro needs espeak-ng (brew install espeak-ng) and
downloads its weights (~330 MB) from Hugging Face on first use.
"""
import argparse
import bisect
import concurrent.futures
import contextlib
import io
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1920, 1080, 25
SR = 24000                  # narration sample rate (Kokoro's native rate)
OUT_SR = 48000              # AAC sample rate in the mp4
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"   # single-pass EBU R128 normalisation
LEAD, TAIL = 0.3, 0.6       # silence before / after each scene's narration (s)
FADE_FRAMES = 8             # cross-fade into each scene
TYPE_CPS = 40               # typewriter speed, characters per second
CAPTION_CHARS = 84          # max characters per subtitle cue
MARGIN = 110
OPENROUTER_URL = "https://openrouter.ai/api/v1"

THEME = {
    "bg": (247, 246, 242), "fg": (28, 30, 34), "muted": (104, 108, 116),
    "accent": (37, 99, 235), "rule": (214, 212, 205), "stripe": (238, 236, 230),
    "hi": (219, 232, 254), "card": (255, 255, 255),
    "term_bg": (24, 26, 31), "term_bar": (44, 47, 54), "term_fg": (210, 214, 220),
    "term_prompt": (125, 211, 252), "term_cmd": (134, 239, 172),
}
SANS = "/System/Library/Fonts/HelveticaNeue.ttc"   # index 0 regular, 1 bold, 10 medium
MONO = "/System/Library/Fonts/Menlo.ttc"           # index 0 regular, 1 bold
_fonts = {}


def font(size: int, weight: str = "regular", mono: bool = False) -> ImageFont.FreeTypeFont:
    idx = {"regular": 0, "bold": 1, "medium": 1 if mono else 10}[weight]
    key = (size, idx, mono)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(MONO if mono else SANS, size, index=idx)
    return _fonts[key]


# ---------------------------------------------------------------- script loading

def substitute(obj, values: dict):
    """Replace {{KEY}} in every string of a nested JSON object."""
    if isinstance(obj, str):
        return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values.get(m.group(1), m.group(0))), obj)
    if isinstance(obj, list):
        return [substitute(x, values) for x in obj]
    if isinstance(obj, dict):
        return {k: substitute(v, values) for k, v in obj.items()}
    return obj


def load_script(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return substitute({k: v for k, v in raw.items() if k != "values"}, raw.get("values", {}))


def leftover_placeholders(script: dict) -> list[str]:
    text = json.dumps(script["scenes"], ensure_ascii=False)
    return sorted(set(re.findall(r"\[FINAL[^\]]*\]|\{\{\w+\}\}", text)))


# ---------------------------------------------------------------- narration

def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text.strip()) if s.strip()]


def spoken(text: str, pronounce: dict) -> str:
    for word in sorted(pronounce, key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(word)}(?![\w-])", pronounce[word], text)
    return text


def say_to_pcm(text: str, voice: str, rate: int, workdir: str, tag: str) -> bytes:
    """Synthesise one sentence with macOS `say`; return mono 16-bit PCM at SR."""
    txt, wav_path = os.path.join(workdir, f"{tag}.txt"), os.path.join(workdir, f"{tag}.wav")
    with open(txt, "w") as f:
        f.write(text)
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-f", txt, "-o", wav_path,
                    "--file-format=WAVE", f"--data-format=LEI16@{SR}"], check=True)
    with wave.open(wav_path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.readframes(w.getnframes())


def kokoro_to_pcm(texts: list[str], voice: str, speed: float) -> list[bytes]:
    """Synthesise each text with Kokoro-82M on the CPU; return mono 16-bit PCM at SR per text.
    The voice prefix picks the accent: a* American, b* British (e.g. bf_emma, am_michael)."""
    import warnings

    import numpy as np
    warnings.filterwarnings("ignore", category=UserWarning)   # torch RNN/weight-norm noise
    warnings.filterwarnings("ignore", category=FutureWarning)
    from kokoro import KPipeline
    pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")
    out = []
    for text in texts:
        chunks = [r.audio.numpy() for r in pipe(text, voice=voice, speed=speed) if r.audio is not None]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        out.append((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return out


def openrouter_style(model: str, style: str | None) -> dict:
    """Provider passthrough carrying a style prompt, per OpenRouter's TTS guide; {} if unsupported."""
    if not style:
        return {}
    if model.startswith("google/"):
        return {"provider": {"options": {"google-ai-studio": {"speech_metadata": {"style": style}}}}}
    if model.startswith("openai/"):
        return {"provider": {"options": {"openai": {"instructions": style}}}}
    print(f"note: no style prompt for {model} via OpenRouter; --style ignored", flush=True)
    return {}


def openrouter_cost(key: str, gen_id: str | None) -> float | None:
    """Look up a generation's cost; the stats lag the request by a moment, so poll briefly."""
    import time

    import httpx
    for _ in range(6) if gen_id else ():
        time.sleep(1.5)
        r = httpx.get(f"{OPENROUTER_URL}/generation", params={"id": gen_id},
                      headers={"Authorization": f"Bearer {key}"}, timeout=30)
        if r.status_code == 200:
            return r.json()["data"].get("total_cost")
    return None


def openrouter_to_pcm(texts: list[str], model: str, voice: str | None, style: str | None,
                      workdir: str) -> list[bytes]:
    """One OpenRouter /audio/speech request per text (MP3, kept as workdir/scene_NN.mp3),
    decoded with ffmpeg to mono 16-bit PCM at SR. Reads OPENROUTER_API_KEY from the environment."""
    from openai import APIStatusError, OpenAI, omit
    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("OPENROUTER_API_KEY is not set")
    client = OpenAI(base_url=OPENROUTER_URL, api_key=key, max_retries=5)  # SDK retries 429/5xx with backoff
    extra, out, total = openrouter_style(model, style), [], 0.0
    for i, text in enumerate(texts):
        try:
            r = client.audio.speech.with_raw_response.create(
                model=model, voice=voice or omit, input=text, response_format="mp3", extra_body=extra or None)
        except APIStatusError as e:  # 402 no credits, 404 unknown model, ...; retries already spent
            sys.exit(f"OpenRouter {model}: HTTP {e.status_code}: {e.message}")
        mp3 = r.content
        with open(os.path.join(workdir, f"scene_{i:02d}.mp3"), "wb") as f:
            f.write(mp3)
        cost = openrouter_cost(key, r.headers.get("x-generation-id"))
        total += cost or 0.0
        print(f"  scene {i + 1}: {len(mp3) / 1e3:.0f} kB mp3, cost "
              f"{'unknown' if cost is None else f'${cost:.5f}'}", flush=True)
        out.append(subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1",
                                   "-ar", str(SR), "pipe:1"], input=mp3, capture_output=True, check=True).stdout)
    print(f"openrouter cost: ${total:.5f} ({model})", flush=True)
    return out


def split_pcm(pcm: bytes, parts: list[str]) -> list[bytes]:
    """Cut one scene's PCM into per-sentence pieces in proportion to their length (for captions)."""
    n, total, cuts, acc = len(pcm) // 2, sum(len(p) for p in parts), [0], 0
    for p in parts:
        acc += len(p)
        cuts.append(round(n * acc / total))
    return [pcm[2 * a:2 * b] for a, b in itertools.pairwise(cuts)]


def synthesise(script: dict, workdir: str, tts: str, model: str | None = None,
               style: str | None = None) -> list[list[tuple[str, bytes]]]:
    """Per scene, a list of (caption text, pcm) per sentence."""
    voice, pron = script["voice"], script.get("pronounce", {})
    jobs = [(si, ti, s) for si, sc in enumerate(script["scenes"])
            for ti, s in enumerate(sentences(sc["narration"]))]
    if tts == "openrouter":  # one request per scene keeps the prosody continuous
        scene_pcm = openrouter_to_pcm([spoken(sc["narration"], pron) for sc in script["scenes"]],
                                      model, voice, style, workdir)
        return [list(zip(ss, split_pcm(pcm, ss))) for pcm, ss in
                zip(scene_pcm, (sentences(sc["narration"]) for sc in script["scenes"]))]
    if tts == "kokoro":
        pcms = kokoro_to_pcm([spoken(s, pron) for _, _, s in jobs], voice, script.get("speed", 1.0))
    else:  # `say` is single-threaded per call, so run sentences in parallel
        rate = script.get("rate", 178)
        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            pcms = list(ex.map(lambda j: say_to_pcm(spoken(j[2], pron), voice, rate, workdir,
                                                    f"s{j[0]:02d}_{j[1]:02d}"), jobs))
    out = [[] for _ in script["scenes"]]
    for (si, _, s), pcm in zip(jobs, pcms):
        out[si].append((s, pcm))
    return out


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * round(seconds * SR)


def chunk_caption(text: str) -> list[str]:
    """Split a sentence into roughly equal cues of at most about CAPTION_CHARS characters."""
    target = len(text) / math.ceil(len(text) / CAPTION_CHARS)
    words, chunks, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > target + 8:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return chunks + ([cur] if cur else [])


def srt_time(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


# ---------------------------------------------------------------- slide drawing

def wrap(text: str, fnt, max_w: float) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        trial = f"{cur} {w}".strip()
        if cur and fnt.getlength(trial) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def draw_bullets(d, spec, x, y, w, _h) -> int:
    size = spec.get("size", 36)
    f, lh, y0 = font(size), round(size * 1.3), y
    for item in spec["bullets"]:
        d.ellipse([x + 4, y + size * 0.42, x + 4 + size * 0.3, y + size * 0.72], fill=THEME["accent"])
        for line in wrap(item, f, w - size * 1.1):
            d.text((x + size * 1.1, y), line, font=f, fill=THEME["fg"])
            y += lh
        y += round(size * 0.55)
    return y - y0


def draw_note(d, spec, x, y, w, _h) -> int:
    size = spec.get("size", 26)
    f, lh, y0 = font(size), round(size * 1.35), y
    for line in wrap(spec["note"], f, w):
        d.text((x, y), line, font=f, fill=THEME["muted"])
        y += lh
    return y - y0


def draw_table(d, spec, x, y, w, _h) -> int:
    t = spec["table"]
    size = t.get("size", 30)
    rows, align, hi = [t["header"]] + t["rows"], t.get("align", "l" * len(t["header"])), set(t.get("highlight", []))
    while True:  # shrink the type until the columns fit the width
        f, fb, pad = font(size), font(size, "bold"), round(size * 0.7)
        col_w = [max(fb.getlength(r[c]) for r in rows) + 2 * pad for c in range(len(t["header"]))]
        if sum(col_w) <= w or size <= 16:
            break
        size -= 2
    col_w[0] += max(0, w - sum(col_w))  # first column takes the slack
    rh = round(size * 1.85)
    for ri, row in enumerate(rows):
        top = y + ri * rh
        if ri == 0:
            d.rectangle([x, top, x + w, top + rh], fill=THEME["fg"])
        elif ri - 1 in hi:
            d.rectangle([x, top, x + w, top + rh], fill=THEME["hi"])
        elif ri % 2 == 0:
            d.rectangle([x, top, x + w, top + rh], fill=THEME["stripe"])
        cx = x
        for c, cell in enumerate(row):
            fnt = fb if ri == 0 or ri - 1 in hi else f
            colour = THEME["bg"] if ri == 0 else THEME["fg"]
            tx = cx + pad if align[c] == "l" else cx + col_w[c] - pad - fnt.getlength(cell)
            d.text((tx, top + rh / 2), cell, font=fnt, fill=colour, anchor="lm")
            cx += col_w[c]
    d.line([x, y + len(rows) * rh, x + w, y + len(rows) * rh], fill=THEME["rule"], width=2)
    return len(rows) * rh


def draw_code(d, spec, x, y, w, _h) -> int:
    size = spec.get("size", 26)
    f, lh, pad = font(size, mono=True), round(size * 1.6), round(size * 0.9)
    lines = [ln for raw in spec["code"] for ln in wrap_mono(raw, int((w - 2 * pad) / f.getlength("M")) - 2)]
    hgt = len(lines) * lh + 2 * pad
    d.rounded_rectangle([x, y, x + w, y + hgt], radius=18, fill=THEME["term_bg"])
    for i, ln in enumerate(lines):
        prefix = "$ " if ln in spec["code"] else "  "
        d.text((x + pad, y + pad + i * lh), prefix, font=f, fill=THEME["term_prompt"])
        d.text((x + pad + f.getlength("$ "), y + pad + i * lh), ln, font=f, fill=THEME["term_cmd"])
    return hgt


def wrap_mono(text: str, cols: int) -> list[str]:
    return [text[i:i + cols] for i in range(0, max(len(text), 1), cols)]


def draw_image(img, d, spec, x, y, w, h) -> int:
    path = os.path.join(ROOT, spec["image"])
    if not os.path.exists(path):
        d.rectangle([x, y, x + w, y + h], outline=THEME["accent"], width=3)
        d.text((x + w / 2, y + h / 2), f"[missing image: {spec['image']}]", font=font(28),
               fill=THEME["accent"], anchor="mm")
        return h
    pic = Image.open(path).convert("RGB")
    pad = 16
    scale = min((w - 2 * pad) / pic.width, (h - 2 * pad) / pic.height)
    pic = pic.resize((round(pic.width * scale), round(pic.height * scale)), Image.LANCZOS)
    cw, ch = pic.width + 2 * pad, pic.height + 2 * pad
    d.rounded_rectangle([x, y, x + cw, y + ch], radius=14, fill=THEME["card"], outline=THEME["rule"], width=2)
    img.paste(pic, (x + pad, y + pad))
    return ch


def draw_blocks(img, d, blocks, x, y, w, bottom) -> None:
    for b in blocks:
        if "bullets" in b:
            used = draw_bullets(d, b, x, y, w, bottom - y)
        elif "table" in b:
            used = draw_table(d, b, x, y, w, bottom - y)
        elif "note" in b:
            used = draw_note(d, b, x, y, w, bottom - y)
        elif "code" in b:
            used = draw_code(d, b, x, y, w, bottom - y)
        elif "image" in b:
            used = draw_image(img, d, b, x, y, w, bottom - y)
        else:
            raise ValueError(f"unknown block {list(b)}")
        y += used + 36


def draw_frame_chrome(slide: dict, index: int, total: int) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    """Background, kicker, title, rule and footer. Returns the y where content starts."""
    img = Image.new("RGB", (W, H), THEME["bg"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 14, H], fill=THEME["accent"])
    y = 72
    if slide.get("kicker"):
        d.text((MARGIN, y), slide["kicker"].upper(), font=font(28, "bold"), fill=THEME["accent"])
        y += 52
    for line in wrap(slide["title"], font(66, "bold"), W - 2 * MARGIN):
        d.text((MARGIN, y), line, font=font(66, "bold"), fill=THEME["fg"])
        y += 80
    if slide.get("subtitle"):
        d.text((MARGIN, y + 4), slide["subtitle"], font=font(30), fill=THEME["muted"])
        y += 46
    y += 18
    d.line([MARGIN, y, W - MARGIN, y], fill=THEME["rule"], width=3)
    foot = font(22)
    d.text((MARGIN, H - 56), "gibc-v2-lm · 49.8M-parameter LM trained from scratch", font=foot, fill=THEME["muted"])
    d.text((W - MARGIN, H - 56), f"{index + 1} / {total}", font=foot, fill=THEME["muted"], anchor="ra")
    return img, d, y + 44


def render_slide(slide: dict, index: int, total: int) -> Image.Image:
    img, d, top = draw_frame_chrome(slide, index, total)
    bottom = H - 96
    if "columns" in slide:
        gap = 70
        widths = slide.get("widths", [1 / len(slide["columns"])] * len(slide["columns"]))
        avail, x = W - 2 * MARGIN - gap * (len(widths) - 1), MARGIN
        for col, frac in zip(slide["columns"], widths):
            draw_blocks(img, d, col, x, top, round(avail * frac), bottom)
            x += round(avail * frac) + gap
    else:
        draw_blocks(img, d, slide.get("blocks", []), MARGIN, top, W - 2 * MARGIN, bottom)
    return img


# ---------------------------------------------------------------- demo (typewriter)

def run_demo(ckpt: str, tok_path: str, cfg: dict) -> list[dict]:
    """Sample each prompt with scripts/demo.py's generator; returns [{prompt, completion}]."""
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import demo  # noqa: E402  (scripts/demo.py)

    torch.manual_seed(cfg.get("seed", 0))
    model, tok = demo.load_model(ckpt, demo.pick_device()), Tokenizer.from_file(tok_path)
    out = []
    for p in cfg["prompts"]:
        with contextlib.redirect_stdout(io.StringIO()):
            text = demo.generate(model, tok, p, cfg.get("max_new", 40), cfg.get("temperature", 0.8),
                                 cfg.get("top_k", 50))
        out.append({"prompt": p, "completion": text})
    return out


def tidy(completion: str) -> str:
    """Collapse blank runs and cut at the last sentence end, if that keeps most of the text."""
    text = re.sub(r"\n{2,}", "\n", completion).rstrip()
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)]
    return text[:ends[-1]] if ends and ends[-1] >= 0.5 * len(text) else text + " ..."


class Typewriter:
    """Terminal panel whose text is revealed character by character on a schedule.
    Completions are shown via tidy(): cut at the last full sentence; demo_output.json keeps them raw."""

    def __init__(self, base: Image.Image, top: int, ckpt: str, tok: str, samples: list[dict]):
        self.base, self.f = base, font(27, mono=True)
        self.fb = font(27, "bold", mono=True)
        self.box = (MARGIN, top, W - MARGIN, H - 96)
        self.pad, self.lh = 30, 38
        self.cw = self.f.getlength("M")
        self.cols = int((self.box[2] - self.box[0] - 2 * self.pad) / self.cw)
        self.rows = int((self.box[3] - self.box[1] - 56 - 2 * self.pad) / self.lh)
        cmd = (f"$ uv run python scripts/demo.py --ckpt {short_path(ckpt)} "
               f"--tok {os.path.basename(tok)} --prompt ...\n")
        self.text, self.kind, self.times = "", [], []
        self._add(cmd, "cmd", None)
        t = 0.4
        for s in samples:
            t = self._add("\n> ", "prompt", t, instant=True)
            t = self._add(s["prompt"], "prompt", t)
            t = self._add(tidy(s["completion"]), "gen", t + 0.5)
            t = self._add("\n", "gen", t, instant=True) + 1.0
        self.duration = t + 1.2
        self.lines = self._layout()

    def _add(self, s: str, kind: str, t, instant: bool = False) -> float:
        for ch in s:
            self.text += ch
            self.kind.append(kind)
            self.times.append(-1.0 if t is None else t)
            if t is not None and not instant:
                t += 1 / TYPE_CPS
        return t if t is not None else 0.0

    def _layout(self) -> list[tuple[int, int]]:
        """Visual lines as (start, end) indices into self.text, word-wrapped to self.cols."""
        lines, start = [], 0
        for logical in self.text.split("\n"):
            end_all = start + len(logical)
            s = start
            while end_all - s > self.cols:
                cut = self.text.rfind(" ", s, s + self.cols)
                cut = cut + 1 if cut > s else s + self.cols
                lines.append((s, cut))
                s = cut
            lines.append((s, end_all))
            start = end_all + 1
        return lines

    def frame(self, t: float) -> Image.Image:
        k = bisect.bisect_right(self.times, t)
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        x0, y0, x1, _ = self.box
        d.rounded_rectangle(self.box, radius=20, fill=THEME["term_bg"])
        d.rounded_rectangle([x0, y0, x1, y0 + 56], radius=20, fill=THEME["term_bar"])
        d.rectangle([x0, y0 + 36, x1, y0 + 56], fill=THEME["term_bar"])
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([x0 + 24 + i * 34, y0 + 18, x0 + 44 + i * 34, y0 + 38], fill=c)
        d.text(((x0 + x1) / 2, y0 + 28), "demo.py — local, Apple MPS", font=font(24), fill=THEME["muted"], anchor="mm")
        visible = [ln for ln in self.lines if ln[0] < k or (ln[0] == ln[1] and ln[0] <= k)]
        visible = visible[-self.rows:]
        cursor = None
        for row, (s, e) in enumerate(visible):
            y = y0 + 56 + self.pad + row * self.lh
            e = min(e, k)
            for i in range(s, e):
                colour, fnt = {"cmd": (THEME["term_cmd"], self.f), "prompt": (THEME["term_prompt"], self.fb),
                               "gen": (THEME["term_fg"], self.f)}[self.kind[i]]
                d.text((x0 + self.pad + (i - s) * self.cw, y), self.text[i], font=fnt, fill=colour)
            cursor = (x0 + self.pad + (e - s) * self.cw, y)
        if cursor and int(t * 2) % 2 == 0:
            d.rectangle([cursor[0], cursor[1] + 3, cursor[0] + self.cw * 0.6, cursor[1] + 32], fill=THEME["term_fg"])
        return img


def short_path(p: str) -> str:
    parts = os.path.normpath(p).split(os.sep)
    return os.path.join(*parts[-2:]) if len(parts) > 1 else p


# ---------------------------------------------------------------- assembly

def encode(scenes_frames, audio_path: str, out: str) -> None:
    """scenes_frames: list of (n_frames, frame_fn(i) -> Image). Streams raw RGB into ffmpeg."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", audio_path, "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-af", LOUDNORM, "-c:a", "aac", "-b:a", "160k", "-ar", str(OUT_SR), "-ac", "2", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    prev = None
    for n, frame_fn in scenes_frames:
        static = getattr(frame_fn, "static", None)
        static_bytes = static.tobytes() if static is not None else None
        for i in range(n):
            if prev is not None and i < FADE_FRAMES:
                cur = static if static is not None else frame_fn(i)
                proc.stdin.write(Image.blend(prev, cur, (i + 1) / (FADE_FRAMES + 1)).tobytes())
            else:
                proc.stdin.write(static_bytes if static_bytes else frame_fn(i).tobytes())
        prev = static if static is not None else frame_fn(n - 1)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")


def static_frames(img: Image.Image):
    fn = lambda i: img  # noqa: E731
    fn.static = img
    return fn


def probe(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration,size:stream=codec_type,codec_name,width,height,sample_rate,channels",
                        "-of", "json", path], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint for the live demo (final.pt)")
    ap.add_argument("--tok", required=True, help="tokenizer.json")
    ap.add_argument("--script", default=os.path.join(ROOT, "video", "script.json"))
    ap.add_argument("--out", help="default video/gibc_demo.mp4, or video/voice_sample.mp4 with --sample")
    ap.add_argument("--tts", choices=["kokoro", "say", "openrouter"], default="kokoro", help="narration backend")
    ap.add_argument("--voice", help="override script.json's voice (kokoro, e.g. bf_emma, bm_george, af_heart) "
                                    "or say_voice (say, see `say -v '?'`); openrouter: the model's voice id")
    ap.add_argument("--model", help="openrouter TTS model id, e.g. google/gemini-3.8-flash-tts")
    ap.add_argument("--style", help="openrouter style prompt (Google Gemini and OpenAI TTS models only)")
    ap.add_argument("--reuse-demo", action="store_true", help="reuse video/demo_output.json, skip sampling")
    ap.add_argument("--sample", action="store_true",
                    help="render only the first scene, to judge a voice (slides and captions.srt untouched)")
    args = ap.parse_args()
    args.out = args.out or os.path.join(ROOT, "video", "voice_sample.mp4" if args.sample else "gibc_demo.mp4")
    if args.tts == "openrouter" and not args.model:
        sys.exit("--tts openrouter needs --model")
    needed = ("ffmpeg", "ffprobe") + {"say": ("say",), "kokoro": ("espeak-ng",)}.get(args.tts, ())
    for tool in needed:
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")

    script = load_script(args.script)
    if args.sample:
        script["scenes"] = script["scenes"][:1]
    default_voice = {"kokoro": script.get("voice", "bf_emma"), "say": script.get("say_voice", "Samantha")}.get(args.tts)
    script["voice"] = args.voice or default_voice
    vdir = os.path.dirname(os.path.abspath(args.out))
    slide_dir = os.path.join(vdir, "slides")
    os.makedirs(slide_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="gibc_video_")

    demo_json = os.path.join(vdir, "demo_output.json")
    if not any(sc.get("type") == "demo" for sc in script["scenes"]):
        samples = []
    elif args.reuse_demo and os.path.exists(demo_json):
        with open(demo_json) as f:
            samples = json.load(f)["samples"]
    else:
        print("sampling demo prompts ...", flush=True)
        samples = run_demo(args.ckpt, args.tok, script["demo"])
        with open(demo_json, "w") as f:
            json.dump({"ckpt": args.ckpt, "demo": script["demo"], "samples": samples}, f, indent=2)

    print(f"synthesising narration with {args.tts}{f' {args.model}' if args.model else ''}, "
          f"voice {script['voice'] or 'default'} ...", flush=True)
    speech = synthesise(script, work, args.tts, args.model, args.style)
    if args.sample and args.tts == "openrouter":  # keep the untouched MP3 next to the clip
        shutil.copy(os.path.join(work, "scene_00.mp3"), os.path.splitext(args.out)[0] + ".mp3")

    scenes, n_total = script["scenes"], len(script["scenes"])
    audio, cues, plan, t0 = bytearray(), [], [], 0.0
    for idx, (sc, clips) in enumerate(zip(scenes, speech)):
        base = render_slide(sc["slide"], idx, n_total)
        speech_s = sum(len(p) / 2 / SR for _, p in clips)
        dur = LEAD + speech_s + TAIL
        if sc.get("type") == "demo":
            _, _, top = draw_frame_chrome(sc["slide"], idx, n_total)
            tw = Typewriter(base, top, args.ckpt, args.tok, samples)
            dur = max(dur, tw.duration)
            frame_fn = lambda i, tw=tw: tw.frame(i / FPS)  # noqa: E731
            tw.frame(tw.duration).save(os.path.join(slide_dir, f"{idx + 1:02d}_{sc['id']}.png"))
        else:
            frame_fn = static_frames(base)
            if not args.sample:
                base.save(os.path.join(slide_dir, f"{idx + 1:02d}_{sc['id']}.png"))
        n_frames = math.ceil(dur * FPS)
        plan.append((n_frames, frame_fn))
        # audio: lead silence, sentences back to back, pad to the exact scene length
        scene_audio, t = bytearray(silence(LEAD)), t0 + LEAD
        for text, pcm in clips:
            d = len(pcm) / 2 / SR
            pieces = chunk_caption(text)
            total_chars = sum(len(p) for p in pieces)
            ct = t
            for p in pieces:
                pd = d * len(p) / total_chars
                cues.append((ct, ct + pd, p))
                ct += pd
            scene_audio += pcm
            t += d
        scene_audio += silence(n_frames / FPS - len(scene_audio) / 2 / SR)
        audio += scene_audio
        t0 += n_frames / FPS

    wav_path = os.path.join(work, "narration.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(audio))
    srt_path = os.path.splitext(args.out)[0] + ".srt" if args.sample else os.path.join(vdir, "captions.srt")
    with open(srt_path, "w") as f:
        for i, (a, b, text) in enumerate(cues, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")

    print(f"encoding {sum(n for n, _ in plan)} frames ({t0:.1f} s) ...", flush=True)
    encode(plan, wav_path, args.out)
    shutil.rmtree(work, ignore_errors=True)

    info = probe(args.out)
    dur, size = float(info["format"]["duration"]), int(info["format"]["size"])
    kinds = [s["codec_type"] for s in info["streams"]]
    words = sum(len(sc["narration"].split()) for sc in scenes)
    print(f"wrote {args.out}: {dur:.1f} s, {size / 1e6:.1f} MB, streams {kinds}, {words} narration words")
    print(f"captions: {srt_path} ({len(cues)} cues); slides: {slide_dir}")
    if not args.sample and not 150 <= dur <= 300:
        print(f"WARNING: duration {dur:.1f} s is outside 150-300 s")
    if "audio" not in kinds:
        print("WARNING: no audio stream")
    left = leftover_placeholders(script)
    if left:
        print("placeholders still to fill in video/script.json 'values':", *left, sep="\n  ")


if __name__ == "__main__":
    main()
