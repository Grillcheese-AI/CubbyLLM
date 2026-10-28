"""pod_remote -- where a run's truth lives between pods (order 0045): the inputs, the newest checkpoint slot, the
metrics, the state file. Two remotes with one face:

- ``FolderRemote``: a local folder (the smoke test, or a mounted volume);
- ``HfRemote``: a private Hugging Face dataset repo named by ``CB_POD_REPO``, the token from ``HF_TOKEN`` (never
  written anywhere; redacted from what this module prints).

``put`` uploads one file (a background ``Uploader`` runs them off the training thread), ``get`` downloads one,
checked by sha256 when the manifest knows it; both resume (a partial download is resumed by huggingface_hub, a
retried upload is a retry). The state file ``pod_state.json`` is small JSON: ``read_state`` / ``write_state``.

Wired: WIRED (``run_pod.py``, ``pod_phases.py``).
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import threading
import time

__wiring__ = "WIRED"

STATE = "pod_state.json"


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def redact(text: str) -> str:
    """A token never reaches a log: anything that looks like one is replaced."""
    tok = os.environ.get("HF_TOKEN", "")
    out = text.replace(tok, "<HF_TOKEN>") if tok else text
    import re
    return re.sub(r"hf_[A-Za-z0-9]{20,}", "<HF_TOKEN>", out)


class FolderRemote:
    """A folder standing in for the repo."""
    kind = "folder"

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def name(self) -> str:
        return self.root

    def exists(self, rel: str) -> bool:
        return os.path.exists(os.path.join(self.root, rel))

    def put(self, local: str, rel: str) -> None:
        dst = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copyfile(local, dst + ".part")
        os.replace(dst + ".part", dst)

    def get(self, rel: str, local: str) -> str:
        src = os.path.join(self.root, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(rel)
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        shutil.copyfile(src, local + ".part")
        os.replace(local + ".part", local)
        return local

    def listing(self) -> list[str]:
        out = []
        for d, _ds, fs in os.walk(self.root):
            for f in fs:
                out.append(os.path.relpath(os.path.join(d, f), self.root).replace(os.sep, "/"))
        return sorted(out)


class HfRemote:
    """A private dataset repo. The token comes from the environment and goes nowhere else."""
    kind = "hf"

    def __init__(self, repo: str, token: str | None = None) -> None:
        from huggingface_hub import HfApi
        self.repo = repo
        self.api = HfApi(token=token or os.environ.get("HF_TOKEN") or None)
        self._files: set | None = None

    def name(self) -> str:
        return self.repo

    def ensure(self) -> None:
        self.api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)

    def listing(self) -> list[str]:
        self._files = set(self.api.list_repo_files(self.repo, repo_type="dataset"))
        return sorted(self._files)

    def exists(self, rel: str) -> bool:
        if self._files is None:
            self.listing()
        return rel in (self._files or set())

    def put(self, local: str, rel: str) -> None:
        for attempt in range(3):
            try:
                self.api.upload_file(path_or_fileobj=local, path_in_repo=rel, repo_id=self.repo, repo_type="dataset")
                if self._files is not None:
                    self._files.add(rel)
                return
            except Exception as e:                   # noqa: BLE001 -- a retry, then the error, redacted
                if attempt == 2:
                    raise RuntimeError(redact(f"upload of {rel} failed: {e}")) from None
                time.sleep(5 * (attempt + 1))

    def get(self, rel: str, local: str) -> str:
        from huggingface_hub import hf_hub_download
        d = os.path.dirname(local) or "."
        os.makedirs(d, exist_ok=True)
        got = hf_hub_download(self.repo, rel, repo_type="dataset", local_dir=d + "/_hf", token=self.api.token)
        shutil.move(got, local)
        return local


def open_remote(spec: str | None):
    """``folder:<path>`` or a bare path -> FolderRemote; ``hf:<repo>`` or ``CB_POD_REPO`` -> HfRemote."""
    spec = spec or (f"hf:{os.environ['CB_POD_REPO']}" if os.environ.get("CB_POD_REPO") else None)
    if not spec:
        raise SystemExit("no remote: set CB_POD_REPO (a private dataset repo) or pass --remote folder:<dir>")
    if spec.startswith("hf:"):
        return HfRemote(spec[3:])
    return FolderRemote(spec[7:] if spec.startswith("folder:") else spec)


def fetch(remote, rel: str, local: str, want_sha: str | None = None) -> str:
    """`rel` into `local`, skipped when a file of the right hash is already there; checked after."""
    if os.path.exists(local) and (want_sha is None or sha256(local) == want_sha):
        return local
    remote.get(rel, local)
    if want_sha is not None and sha256(local) != want_sha:
        os.remove(local)
        raise RuntimeError(f"{rel}: sha256 mismatch after download")
    return local


def read_state(remote, local_dir: str) -> dict:
    """The state file: the remote's copy wins over a local one (a new pod starts from the remote)."""
    local = os.path.join(local_dir, STATE)
    if remote.exists(STATE):
        try:
            remote.get(STATE, local)
        except Exception:                             # noqa: BLE001 -- the local copy, then
            pass
    if os.path.exists(local):
        try:
            return json.load(open(local, encoding="utf-8"))
        except ValueError:
            pass
    return {"phase": None, "done": [], "arm": None, "last_step_uploaded": {}, "preemptions": 0, "card": None}


def write_state(remote, local_dir: str, state: dict) -> None:
    local = os.path.join(local_dir, STATE)
    os.makedirs(local_dir, exist_ok=True)
    state["updated"] = time.strftime("%F %T")
    with open(local + ".tmp", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(local + ".tmp", local)
    remote.put(local, STATE)


class Uploader:
    """Uploads queued off the training thread, in order; ``wait`` drains the queue (a phase's end, the final)."""

    def __init__(self, remote) -> None:
        self.remote, self.q, self.errors = remote, queue.Queue(), []
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            local, rel = item
            try:
                self.remote.put(local, rel)
            except Exception as e:                   # noqa: BLE001 -- kept, reported by `wait`; the run goes on
                self.errors.append(redact(f"{rel}: {e}"))
            finally:
                self.q.task_done()

    def put(self, local: str, rel: str) -> None:
        self.q.put((local, rel))

    def wait(self) -> list[str]:
        self.q.join()
        return list(self.errors)

    def close(self) -> None:
        self.q.put(None)
        self._t.join(timeout=60)


__all__ = ["FolderRemote", "HfRemote", "Uploader", "fetch", "open_remote", "read_state", "redact", "sha256", "write_state"]
