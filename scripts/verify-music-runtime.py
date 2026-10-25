"""Validate the isolated inference imports, pinned source revision and real CUDA."""

import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import generate_music
from pytorch_wavelets import DWT1DForward, DWT1DInverse

revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
assert revision == "ca1e85fe9430179831e6bc6be790c332190a3866", "源码版本不匹配"
assert torch.cuda.is_available(), "CUDA 不可用"
assert torch.isfinite(torch.ones(32, device="cuda") @ torch.ones(32, device="cuda"))
sample = torch.randn(1, 2, 64, device="cuda")
coefficients = DWT1DForward(J=1, wave="haar").cuda()(sample)
restored = DWT1DInverse(wave="haar").cuda()(coefficients)
assert torch.allclose(sample, restored, atol=1e-5), "CUDA 音质处理组件未通过检查"
evidence = {"revision": revision, "torch": torch.__version__, "cuda_verified": True,
    "gpu": torch.cuda.get_device_name(0), "imports_verified": True, "wavelets_verified": True,
    "environment_purpose": "ACE-Step inference only; separate from speech and RVC dependencies"}
(root / "studio-runtime.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print(json.dumps(evidence, ensure_ascii=False))
