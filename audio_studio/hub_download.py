"""Bounded, resumable Hub transfers with official SHA256 verification."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
from urllib.request import ProxyHandler, Request, build_opener, getproxies


def verified(path: Path, size: int, digest: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == digest


def download_file(repo: str, revision: str, filename: str, target: Path,
                  size: int, digest: str, log=print) -> None:
    """Expose a file only after its official size and SHA256 match."""
    urls = [(f"https://hf-mirror.com/{repo}/resolve/{revision}/{filename}", False),
        (f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", True)]
    if repo.startswith("ACE-Step/"):
        # The author publishes the same weights on ModelScope. The official
        # Hub hash remains authoritative, including when this mirror updates.
        urls.insert(0, (f"https://modelscope.cn/models/{repo}/resolve/master/{filename}", False))
    download_blob(urls, filename, target, size, digest, log)


def download_blob(urls: list[tuple[str, bool]], filename: str, target: Path,
                  size: int, digest: str, log=print) -> None:
    """Resume verified ranges from explicit sources; proxy use is per source."""
    if verified(target, size, digest):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".download")
    markers = target.parent / ".ranges" / digest
    markers.mkdir(parents=True, exist_ok=True)
    if not partial.exists() or partial.stat().st_size != size:
        with partial.open("wb") as stream:
            stream.truncate(size)
        # Markers from a previously missing partial must never skip new bytes.
        for marker in markers.glob("*.ok"):
            marker.unlink()
    block_size = 16 * 1024 * 1024
    ranges = [(start, min(start + block_size, size) - 1) for start in range(0, size, block_size)]

    def fetch(bounds):
        start, end = bounds
        marker = markers / f"{start}.ok"
        if marker.is_file():
            return
        attempts = max(4, len(urls) * 2)
        for attempt in range(attempts):
            source, use_proxy = urls[min(attempt // 2, len(urls) - 1)]
            url = source + ("&" if "?" in source else "?") + f"range_start={start}"
            try:
                opener = build_opener(ProxyHandler(getproxies() if use_proxy else {}))
                request = Request(url, headers={"Range": f"bytes={start}-{end}"})
                with opener.open(request, timeout=60) as response:
                    if size > block_size and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/")):
                        raise RuntimeError("模型下载范围不匹配")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise RuntimeError("模型下载数据不完整")
                with partial.open("r+b") as stream:
                    stream.seek(start)
                    stream.write(data)
                marker.write_text("ok", encoding="ascii")
                return
            except (OSError, RuntimeError):
                if attempt == attempts - 1:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, item) for item in ranges]
        try:
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 16 == 0 or index == len(ranges):
                    log(f"{filename}：{index}/{len(ranges)} 段完成")
        except Exception:
            for future in futures:
                future.cancel()
            raise
    if not verified(partial, size, digest):
        # A subsequent attempt must replace corrupt ranges rather than trust them.
        for marker in markers.glob("*.ok"):
            marker.unlink()
        raise RuntimeError(f"{filename} 的官方 SHA256 校验失败；再次下载将重取数据")
    partial.replace(target)
    log(f"{filename}：官方 SHA256 校验通过")
