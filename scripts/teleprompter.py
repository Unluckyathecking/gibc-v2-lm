"""Teleprompter and per-scene recorder for voicing the demo video yourself.

Usage:
  uv run python scripts/teleprompter.py [--port 8741] [--no-browser]

Serves scripts/teleprompter.html on http://127.0.0.1:<port> (localhost only; browsers allow
the microphone there without https) and opens it, in Chrome if installed. The page shows each
scene's slide and narration from video/script.json, paces the words at a target rate, and
records the mic per scene; each take is POSTed back here and saved as
video/voiceover/scene_NN.<ext> plus scene_NN.json (its length after silence trimming).
Then render with: make_video.py --tts voiceover (see docs/VIDEO_SCRIPT.md).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_video as mv  # noqa: E402  (scripts/make_video.py: script loading, slides, take decoding)

HTML = os.path.join(mv.ROOT, "scripts", "teleprompter.html")
SLIDE_DIR = os.path.join(mv.ROOT, "video", "slides")
MAX_TAKE_BYTES = 200 * 1024 * 1024
AUDIO_TYPES = {"webm": "audio/webm", "wav": "audio/wav", "m4a": "audio/mp4", "mp3": "audio/mpeg",
               "ogg": "audio/ogg"}


def slide_path(i: int, sc: dict) -> str:
    return os.path.join(SLIDE_DIR, f"{i + 1:02d}_{sc['id']}.png")


def ensure_slides(scenes: list[dict]) -> None:
    """Render any missing slide PNG with make_video's renderer (the demo slide without its typewriter)."""
    os.makedirs(SLIDE_DIR, exist_ok=True)
    for i, sc in enumerate(scenes):
        if not os.path.exists(slide_path(i, sc)):
            print(f"rendering missing slide {os.path.basename(slide_path(i, sc))}")
            mv.render_slide(sc["slide"], i, len(scenes)).save(slide_path(i, sc))


class Prompter(BaseHTTPRequestHandler):
    scenes: list[dict] = []
    vo_dir = ""

    # ------------------------------------------------------------ helpers
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _file(self, path: str, ctype: str) -> None:
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _scene_index(self, raw: str | None) -> int | None:
        """1-based scene number from a string, or None if it is not a valid scene."""
        if raw is None or not raw.isdigit() or not 1 <= int(raw) <= len(self.scenes):
            return None
        return int(raw)

    def _take_info(self, n: int) -> dict | None:
        path = mv.find_takes(self.vo_dir, n)[n - 1]
        meta = os.path.join(self.vo_dir, f"scene_{n:02d}.json")
        if path is None or not os.path.exists(meta):
            return None
        with open(meta) as f:
            info = json.load(f)
        return info if info.get("file") == os.path.basename(path) else None

    def log_message(self, fmt, *args) -> None:  # quiet: print only what we choose to
        pass

    # ------------------------------------------------------------ routes
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._file(HTML, "text/html; charset=utf-8")
        elif path == "/api/script":
            self._json({
                "scenes": [{"n": i + 1, "id": sc["id"], "title": sc["slide"].get("title", sc["id"]),
                            "narration": sc["narration"], "demo": sc.get("type") == "demo"}
                           for i, sc in enumerate(self.scenes)],
                "takes": {n: t for n in range(1, len(self.scenes) + 1) if (t := self._take_info(n))},
                "wpm": mv.TARGET_WPM, "lead": mv.LEAD, "tail": mv.TAIL,
                "voiceover_dir": os.path.relpath(self.vo_dir, mv.ROOT),
            })
        elif m := re.fullmatch(r"/slides/(\d+)", path):
            n = self._scene_index(m.group(1))
            if n is None:
                return self._json({"error": "no such scene"}, 404)
            self._file(slide_path(n - 1, self.scenes[n - 1]), "image/png")
        elif m := re.fullmatch(r"/takes/(\d+)", path):
            n = self._scene_index(m.group(1))
            take = mv.find_takes(self.vo_dir, n)[n - 1] if n else None
            if take is None:
                return self._json({"error": "no take for that scene"}, 404)
            self._file(take, AUDIO_TYPES[take.rsplit(".", 1)[1]])
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path != "/api/take":
            return self._json({"error": "not found"}, 404)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        n, ext = self._scene_index(q.get("scene")), q.get("ext", "")
        if n is None or ext not in AUDIO_TYPES:
            return self._json({"error": f"need scene=1..{len(self.scenes)} and ext in {sorted(AUDIO_TYPES)}"}, 400)
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_TAKE_BYTES:
            return self._json({"error": f"body must be 1 byte to {MAX_TAKE_BYTES} bytes, got {length}"}, 400)
        body = self.rfile.read(length)
        try:
            self._json(self._save_take(n, ext, body, q.get("mime", "")))
        except ValueError as e:  # undecodable audio: keep the previous take
            print(f"scene {n}: rejected take: {e}", flush=True)
            self._json({"error": str(e)}, 422)

    def _save_take(self, n: int, ext: str, body: bytes, mime: str) -> dict:
        """Decode-check the upload, then replace scene n's take and write its JSON sidecar."""
        os.makedirs(self.vo_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=f".{ext}", dir=self.vo_dir)
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        try:
            speech_s = len(mv.take_to_pcm(tmp)) / 2 / mv.SR
        except ValueError:
            os.remove(tmp)
            raise ValueError(f"could not decode the {ext} upload ({len(body)} bytes)") from None
        name = f"scene_{n:02d}.{ext}"
        for e in AUDIO_TYPES:  # a retake in another format must not leave the old file behind
            old = os.path.join(self.vo_dir, f"scene_{n:02d}.{e}")
            if e != ext and os.path.exists(old):
                os.remove(old)
        os.chmod(tmp, 0o644)  # mkstemp creates it owner-only
        os.replace(tmp, os.path.join(self.vo_dir, name))
        sc = self.scenes[n - 1]
        words = len(sc["narration"].split())
        info = {"scene": n, "id": sc["id"], "file": name, "mime": mime, "bytes": len(body),
                "duration_s": round(speech_s, 2), "words": words,
                "target_s": round(words / mv.TARGET_WPM * 60, 2),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(os.path.join(self.vo_dir, f"scene_{n:02d}.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"saved {name}: {speech_s:.1f} s of speech (target {info['target_s']:.1f} s)", flush=True)
        return info


def serve(port: int) -> ThreadingHTTPServer:
    """Bind 127.0.0.1 on the first free port from `port` upwards."""
    for p in range(port, port + 20):
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), Prompter)
        except OSError:
            continue
    sys.exit(f"ports {port}-{port + 19} are all in use; pass --port")


def open_browser(url: str) -> None:
    """Chrome if installed (its MediaRecorder is the most reliable), else the default browser."""
    if sys.platform == "darwin" and os.path.exists("/Applications/Google Chrome.app"):
        subprocess.run(["open", "-a", "Google Chrome", url], check=False)
    else:
        webbrowser.open(url)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", type=int, default=8741, help="first port to try; the next free one is used if taken")
    ap.add_argument("--script", default=os.path.join(mv.ROOT, "video", "script.json"))
    ap.add_argument("--voiceover-dir", default=os.path.join(mv.ROOT, "video", "voiceover"))
    ap.add_argument("--no-browser", action="store_true", help="just serve; open the URL yourself")
    args = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (brew install ffmpeg)")

    scenes = mv.load_script(args.script)["scenes"]
    ensure_slides(scenes)
    Prompter.scenes, Prompter.vo_dir = scenes, os.path.abspath(args.voiceover_dir)
    server = serve(args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"teleprompter on {url} ({len(scenes)} scenes; takes go to {Prompter.vo_dir}); Ctrl+C to stop",
          flush=True)
    if not args.no_browser:
        open_browser(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
