"""Recover a failed partial clone using GitHub raw blobs at its pinned commit."""
import argparse
import hashlib
import json
import subprocess
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor
from .data import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy")
    args = parser.parse_args()
    repo = ROOT / "external/D2LS"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    entries = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "HEAD"], text=True).splitlines()
    selected = []
    for line in entries:
        metadata, name = line.split("\t", 1)
        if name.startswith(("network/", "config/", "tools/")) or name in ("train.py", "README.md", "requirements.txt", ".gitignore"):
            selected.append((metadata.split()[2], name))
    selected.sort(key=lambda item: (item[1] not in ("network/models/d2ls.py", "train.py", "network/losses/useful_loss.py"), item[1]))
    def fetch(entry):
        expected, name = entry
        path = repo / name
        url = f"https://raw.githubusercontent.com/XavierJiezou/D2LS/{revision}/{name}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": args.proxy, "http": args.proxy})) if args.proxy else urllib.request.build_opener()
        content = path.read_bytes() if path.is_file() else b""
        def blob_sha(data):
            return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob_sha(content) != expected:
            for attempt in range(3):
                try:
                    with opener.open(url, timeout=25) as response:
                        content = response.read()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1)
        if blob_sha(content) != expected:
            raise ValueError(f"Git blob hash mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"Verified {name}", flush=True)
        return {"path": name, "git_blob": expected}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, selected))
    manifest = ROOT / "external/d2ls_snapshot.json"
    manifest.write_text(json.dumps({"revision": revision, "verified_files": results}, indent=2), encoding="utf-8")
    print(f"Verified {len(results)} original D2LS files at {revision}", flush=True)


if __name__ == "__main__":
    main()
