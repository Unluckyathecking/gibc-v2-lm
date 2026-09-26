"""Modal app "gibc-train": H100 training runs, a server-side driver, smoke tests.

  modal run modal_train.py::smoke --model A --minutes 5
  modal run --detach modal_train.py::launch --group sweep     # or main, or run1,run2
  modal run modal_train.py::status

The driver runs on Modal (CPU), so the sweep survives the local client exiting. Each
train_run resumes from its own latest checkpoint, so a retry continues where it died.
"""
import json
import os
import time
from datetime import datetime, timezone

import modal

from gibc.configs import RUNS, RUNS_DIR, SWEEP_RUNS
from gibc.modal_common import H100, VOLUMES, data_vol, image, runs_vol

app = modal.App("gibc-train")
DAY = 24 * 60 * 60   # Modal's maximum function timeout
MAX_RETRIES = 3      # per run, after the first attempt
POLL_S = 60          # driver poll interval


@app.function(image=image, gpu=H100, volumes=VOLUMES, timeout=DAY)
def train_run(run_name: str) -> dict:
    from gibc.train import train

    data_vol.reload()
    try:
        return train(run_name, on_checkpoint=runs_vol.commit)
    finally:
        runs_vol.commit()  # final status (done or failed) and final.pt


@app.function(image=image, gpu=H100, volumes=VOLUMES, timeout=2 * 60 * 60)
def smoke(model: str = "A", minutes: float = 5.0) -> dict:
    from gibc.train import smoke as run_smoke

    res = run_smoke(model, minutes)
    print(json.dumps(res, indent=2))
    with open(f"{RUNS_DIR}/smoke_{model}.json", "w") as f:
        json.dump(res, f, indent=2)
    runs_vol.commit()
    return res


def group_runs(group: str) -> list[str]:
    if group == "sweep":
        return list(SWEEP_RUNS)
    if group == "main":
        runs = [r for r in RUNS if r.startswith("main_")]
    else:
        runs = [r.strip() for r in group.split(",") if r.strip()]
    unknown = [r for r in runs if r not in RUNS]
    if unknown or not runs:
        raise ValueError(f"bad group {group!r}: unknown runs {unknown}")
    return runs


def _poll(call) -> tuple[str, object]:
    """("running", None) | ("done", result) | ("failed", exception)."""
    try:
        return "done", call.get(timeout=0)
    except TimeoutError:  # modal raises the builtin TimeoutError while still running
        return "running", None
    except Exception as e:
        return "failed", e


def _deterministic(err: Exception) -> bool:
    """Failures a retry would only repeat: NaN loss, bad config or manifest, and data
    exhaustion (MixLoader and the train preflight both say "exhausted"). Other
    RuntimeErrors (CUDA faults, preemption, infra) stay retryable."""
    if isinstance(err, (FloatingPointError, KeyError, ValueError)):
        return True
    return isinstance(err, RuntimeError) and "exhausted" in str(err)


def _on_failure(s: dict, call, err: Exception) -> bool:
    """Record a failed poll on run state s; True if the run should be respawned."""
    s["errors"].append(repr(err))
    try:
        call.cancel()  # the poll failing need not mean the container stopped: never run two
    except Exception as e:
        print(f"[driver] cancel failed: {e!r}")
    if _deterministic(err) or s["attempts"] > MAX_RETRIES:
        s["state"] = "failed"
        return False
    return True


@app.function(image=image, volumes={RUNS_DIR: runs_vol}, timeout=DAY)
def driver(group: str) -> dict:
    runs = group_runs(group)
    path = f"{RUNS_DIR}/drivers/{group.replace(',', '+')}_status.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    calls = {r: train_run.spawn(r) for r in runs}
    runs_state = {r: {"state": "running", "attempts": 1, "errors": []} for r in runs}

    def write():
        with open(path, "w") as f:
            json.dump({"group": group, "updated": datetime.now(timezone.utc).isoformat(),
                       "runs": runs_state}, f, indent=2)
        runs_vol.commit()

    write()
    while any(s["state"] == "running" for s in runs_state.values()):
        time.sleep(POLL_S)
        for r, s in runs_state.items():
            if s["state"] != "running":
                continue
            state, out = _poll(calls[r])
            if state == "done":
                s.update(state="done", result=out)
            elif state == "failed" and _on_failure(s, calls[r], out):
                calls[r] = train_run.spawn(r)  # resumes from its latest checkpoint
                s["attempts"] += 1
            if state != "running":
                print(f"[driver] {r}: {s['state']} (attempt {s['attempts']})")
                write()
    write()
    return runs_state


@app.local_entrypoint()
def launch(group: str = "sweep"):
    runs = group_runs(group)  # fail fast locally on a bad group
    call = driver.spawn(group)
    print(f"driver {call.object_id} spawned for {group}: {runs}")
    print("progress: modal run modal_train.py::status")


def _read(path: str) -> dict | None:
    try:
        return json.loads(b"".join(runs_vol.read_file(path)))
    except FileNotFoundError:
        return None


@app.local_entrypoint()
def status():
    dirs = [e.path for e in runs_vol.listdir("/") if e.type == modal.volume.FileEntryType.DIRECTORY]
    for d in sorted(dirs):
        if d == "drivers":
            for entry in runs_vol.listdir("/drivers"):
                drv = _read(entry.path)
                print(f"{entry.path}: " + json.dumps(
                    {r: [s["state"], s["attempts"]] for r, s in drv["runs"].items()}))
        elif st := _read(f"{d}/status.json"):
            print(f"{d}: {json.dumps(st)}")
