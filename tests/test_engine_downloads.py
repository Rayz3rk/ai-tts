"""Resume interrupted model downloads without exposing corrupt checkpoints."""

import hashlib
from io import BytesIO
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

from audio_studio import engines


def download_fixture(monkeypatch, data: bytes):
    sibling = SimpleNamespace(rfilename="model.bin", size=len(data), lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()))
    api = SimpleNamespace(model_info=Mock(return_value=SimpleNamespace(siblings=[sibling])))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_url=lambda **kwargs: "https://weights.invalid/model.bin"))
    monkeypatch.setattr(engines.time, "sleep", lambda seconds: None)
    return api


class Response(BytesIO):
    status = 206

    def __init__(self, body: bytes, start: int, total: int):
        super().__init__(body)
        self.headers = {"Content-Range": f"bytes {start}-{total - 1}/{total}"}


def test_weight_download_resumes_and_verifies_before_installing(tmp_path, monkeypatch):
    data = b"a real checkpoint byte sequence"
    api = download_fixture(monkeypatch, data)
    (tmp_path / "model.bin.download").write_bytes(data[:7])
    requests = []

    def open_response(request, timeout):
        requests.append(request.get_header("Range"))
        return Response(data[7:], 7, len(data))

    monkeypatch.setattr(engines, "urlopen", open_response)
    engines._download_weight(api, "owner/repo", "revision", "model.bin", tmp_path, Mock())
    assert requests == [f"bytes=7-{len(data) - 1}"]
    assert (tmp_path / "model.bin").read_bytes() == data
    assert not (tmp_path / "model.bin.download").exists()


def test_corrupt_weight_never_becomes_an_available_model(tmp_path, monkeypatch):
    data = b"official bytes"
    api = download_fixture(monkeypatch, data)
    monkeypatch.setattr(engines, "urlopen", lambda *args, **kwargs: Response(b"x" * len(data), 0, len(data)))
    with pytest.raises(RuntimeError, match="SHA256"):
        engines._download_weight(api, "owner/repo", "revision", "model.bin", tmp_path, Mock())
    assert not (tmp_path / "model.bin").exists()


def test_verified_existing_weight_avoids_redundant_transfer(tmp_path, monkeypatch):
    data = b"already downloaded"
    api = download_fixture(monkeypatch, data)
    (tmp_path / "model.bin").write_bytes(data)
    transfer = Mock(side_effect=AssertionError("unexpected network transfer"))
    monkeypatch.setattr(engines, "urlopen", transfer)
    engines._download_weight(api, "owner/repo", "revision", "model.bin", tmp_path, Mock())
    transfer.assert_not_called()
