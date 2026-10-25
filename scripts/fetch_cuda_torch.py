"""Resume the official CUDA wheel in ranges after streaming-install timeouts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_studio.hub_download import download_blob

root = Path(__file__).resolve().parents[1]
filename = "torch-2.7.1+cu128-cp312-cp312-win_amd64.whl"
target = root / "work" / "verified-wheels" / filename
wheel_path = "/cu128/torch-2.7.1%2Bcu128-cp312-cp312-win_amd64.whl"
download_blob([("https://mirrors.nju.edu.cn/pytorch/whl" + wheel_path, False),
    ("https://mirror.sjtu.edu.cn/pytorch-wheels" + wheel_path, False),
    ("https://download-r2.pytorch.org/whl" + wheel_path, True)],
    filename, target, 3273024349,
    "2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30")
print(target, flush=True)
