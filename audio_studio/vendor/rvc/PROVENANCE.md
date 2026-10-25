# RVC neural network sources

`models.py`, `commons.py`, `attentions.py`, `modules.py`, `transforms.py`
and `rmvpe.py` come from the MIT-licensed official
[RVC repository](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI),
tag `2.2.231006`, commit `9f2f0559e6932c10c48642d404e7d2e771d9db43`.
The original paths were `infer/lib/infer_pack/*.py` and `infer/lib/rmvpe.py`.
The original MIT license is preserved in `LICENSE`.

Local adaptations:

- Imports inside the neural network package are relative, so an installed wheel
  does not depend on the legacy Voice-Pro directory or a modified `sys.path`.
- RMVPE uses its native PyTorch float32 path on CPU/CUDA. The optional Intel,
  DirectML and JIT export loaders are not initialized. Checkpoints are read with
  `weights_only=True`.
- Empty RMVPE confidence windows use a bounded denominator when decoding F0.

`audio_studio.svc` supplies application-specific validation, downloading, audio
segmentation and model registration. HuBERT/ContentVec inference follows the
transformers adapter in the official
[Applio repository](https://github.com/IAHispano/Applio/blob/c817f1f83f689ef72e0dfeb1c17ae990b9248506/rvc/lib/utils.py).
Only RVC v2 pitch-conditioned models are exposed by the application.
