"""Fetch RVC weights through a mirror, verifying official Hugging Face LFS hashes.

The normal in-app downloader remains available. This bounded parallel transfer is
useful on connections where the official large-file endpoint is slow.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import ProxyHandler, Request, build_opener, getproxies

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1] / "studio_data" / "models" / "rvc"
FILES = [("IAHispano/Applio", "5c1d7ed4f4cc3c6ee08b6455e499d80221aca85d", "Resources/embedders/contentvec/pytorch_model.bin"),
    *(('lj1995/VoiceConversionWebUI', 'e6d0c1a17da07c33557852f9dfa2bd44cc75737d', name)
      for name in ('rmvpe.pt', 'pretrained_v2/f0G40k.pth', 'pretrained_v2/f0D40k.pth'))]


def main() -> None:
    for scheme, proxy in getproxies().items():
        if scheme in {"http", "https"}:
            os.environ.setdefault(scheme.upper() + "_PROXY", proxy)
    api = HfApi(endpoint="https://huggingface.co")
    metadata = {}
    tasks = []
    valid_targets = set()
    for repo, revision, filename in FILES:
        if (repo, revision) not in metadata:
            metadata[repo, revision] = api.model_info(repo, revision=revision, files_metadata=True, timeout=30)
        sibling = next(s for s in metadata[repo, revision].siblings if s.rfilename == filename)
        target = ROOT / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == sibling.size:
            with target.open('rb') as source:
                if hashlib.file_digest(source, 'sha256').hexdigest() == sibling.lfs.sha256:
                    valid_targets.add(filename)
                    continue
        directory = ROOT / ".mirror-chunks" / sibling.lfs.sha256
        directory.mkdir(parents=True, exist_ok=True)
        for offset in range(0, sibling.size, 8 * 1024 * 1024):
            end = min(offset + 8 * 1024 * 1024, sibling.size) - 1
            tasks.append((repo, revision, filename, sibling, directory, offset, end))

    def fetch(task):
        repo, revision, filename, sibling, directory, offset, end = task
        part = directory / f"{offset:012d}.part"
        if part.is_file() and part.stat().st_size == end - offset + 1:
            return
        url = f"https://hf-mirror.com/{repo}/resolve/{revision}/{filename}"
        for attempt in range(4):
            try:
                request = Request(url, headers={"Range": f"bytes={offset}-{end}"})
                with build_opener(ProxyHandler({})).open(request, timeout=40) as response:
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Mirror returned an unexpected range")
                    data = response.read(end - offset + 2)
                if len(data) != end - offset + 1:
                    raise RuntimeError("Mirror range is truncated")
                temporary = part.with_suffix('.tmp')
                temporary.write_bytes(data)
                temporary.replace(part)
                return
            except (OSError, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f"Verified-size ranges: {index}/{len(futures)}", flush=True)
    evidence = []
    for repo, revision, filename in FILES:
        sibling = next(s for s in metadata[repo, revision].siblings if s.rfilename == filename)
        target = ROOT / filename
        if filename not in valid_targets:
            temporary = target.with_name(target.name + '.mirror-complete')
            directory = ROOT / ".mirror-chunks" / sibling.lfs.sha256
            with temporary.open('wb') as output:
                for part in sorted(directory.glob('*.part')):
                    output.write(part.read_bytes())
            with temporary.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            if temporary.stat().st_size != sibling.size or digest != sibling.lfs.sha256:
                raise RuntimeError(f"Official SHA256 verification failed: {filename}")
            temporary.replace(target)
        with target.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        assert digest == sibling.lfs.sha256 and target.stat().st_size == sibling.size
        evidence.append({"repository": repo, "revision": revision, "file": filename, "size": sibling.size, "sha256": digest})
        print("Official SHA256 verified: " + filename, flush=True)
    (ROOT / 'mirror-verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
