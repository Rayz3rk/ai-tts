"""Standalone ACE-Step inference entry point, executed by its own Python runtime."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


def generate(payload: dict) -> dict:
    """Generate an original vocal song from lyrics and style with the pinned model."""
    runtime, models = Path(payload["runtime_path"]), Path(payload["path"])
    os.environ.update(ACESTEP_CHECKPOINTS_DIR=str(models), ACESTEP_PROJECT_ROOT=str(runtime),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    sys.path.insert(0, str(runtime))
    import numpy as np
    import soundfile as sf
    import torch
    from pytorch_wavelets import DWT1DForward  # Require the intended sampler correction.
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    if not torch.cuda.is_available():
        raise RuntimeError("自动创作运行环境未检测到 CUDA，请运行安装脚本检查显卡组件")
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    class LocalHandler(AceStepHandler):
        def _ensure_models_present(self, *, checkpoint_path, config_path, prefer_source, vae_variant=None):
            # Upstream's default precheck insists on the unused 1.7B LM. This
            # deployment uses the documented 0.6B LM for 12GB cards instead.
            required = [checkpoint_path / config_path / "model.safetensors",
                checkpoint_path / "vae" / "diffusion_pytorch_model.safetensors",
                checkpoint_path / "Qwen3-Embedding-0.6B" / "model.safetensors"]
            if not all(path.is_file() for path in required):
                return "ERROR: 本地自动创作权重不完整，请先下载", False
            return None

    dit = LocalHandler()
    print("加载 ACE-Step 2B turbo；文本编码器使用按需显存调度。", flush=True)
    message, ok = dit.initialize_service(project_root=str(runtime), config_path=payload["dit_model"],
        device="cuda", offload_to_cpu=True, offload_dit_to_cpu=False,
        compile_model=False, quantization=None, use_flash_attention=False)
    if not ok:
        raise RuntimeError(message)
    print("加载 0.6B 音乐语言模型；根据歌词规划旋律与节奏。", flush=True)
    lm = LLMHandler()
    message, ok = lm.initialize(checkpoint_dir=str(models), lm_model_path=payload["lm_model"],
        backend="pt", device="cuda", offload_to_cpu=True, dtype=torch.bfloat16)
    if not ok:
        raise RuntimeError(message)
    params = GenerationParams(caption=payload["style"], lyrics=payload["lyrics"],
        vocal_language=payload["language"], duration=payload["duration"], bpm=payload.get("bpm"),
        inference_steps=8, seed=payload["seed"], thinking=True,
        use_cot_metas=False, use_cot_caption=False, use_cot_lyrics=False, use_cot_language=False)
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False,
        seeds=[payload["seed"]])
    output = Path(payload["output_path"])
    intermediate = output.parent / (output.stem + "-ace-step")
    print("开始生成原创歌曲，输出包含演唱与伴奏。", flush=True)
    result = generate_music(dit, lm, params, config, save_dir=str(intermediate))
    if not result.success or not result.audios:
        raise RuntimeError(result.error or result.status_message or "自动创作未返回音频")
    source = Path(result.audios[0]["path"])
    waveform, rate = sf.read(source, dtype="float32")
    if not np.isfinite(waveform).all() or np.max(np.abs(waveform), initial=0) < 1e-5:
        raise RuntimeError("自动创作返回无效或静音音频")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return {"path": str(output), "sample_rate": rate, "duration_seconds": len(waveform) / rate,
        "engine": "ace_step", "device": "cuda", "seed": payload["seed"],
        "attribution": "AI 音乐生成：ACE-Step 1.5", "contains_accompaniment": True,
        "runtime_revision": payload["revision"], "dit_model": payload["dit_model"],
        "lm_model": payload["lm_model"], "sampler_correction": "DCW double",
        "quality_revision": "ace-step-dcw-1"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = generate(json.loads(args.request.read_text(encoding="utf-8")))
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
