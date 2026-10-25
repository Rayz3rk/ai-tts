"""Local DiffSinger ONNX singing from explicit lyrics, notes and timing.

Uses the author's unmodified Qixuan models for duration, pitch, voice parameters,
acoustics, and the matching OpenVPI neural vocoder. No TTS or pitch-warp fallback.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import time
from urllib.request import Request, getproxies, urlopen
import zipfile

from audio_studio.schemas import SingingScore

BANK = "Qixuan_v2.7.0_DiffSinger_OpenUtau"
VOCODER = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"
SOURCE = "https://github.com/yqzhishen/qixuan-diffsinger"
ATTRIBUTION = "AI 歌声合成：绮萱 v2.7.0 / DiffSinger；声码器：OpenVPI PC-NSF-HiFiGAN 2025.02"
ASSETS = [
    {"name": BANK + ".zip", "url": SOURCE + "/releases/download/v2.7.0/" + BANK + ".zip",
     "size": 420707627, "sha256": "fe8ee7c95883d327a2b7facbbe832a7a53e1344918dc791ca83fc23b08034110"},
    {"name": VOCODER.replace(".onnx", ".oudep"),
     "url": "https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/" + VOCODER.replace(".onnx", ".oudep"),
     "size": 52760610, "sha256": "ba7d43142d41f6900c8264b5662ca7125a50feb8760bb8b9615c61a8f5e6902e"}]
SAMPLE_RATE, HOP = 44100, 512
PHRASE_MARGIN_FRAMES = 13
QUALITY_REVISION = "vowel-anchors-continuous-f0-1"


def model_metadata(root: Path) -> dict:
    root = root.expanduser().resolve()
    return {"engine": "diffsinger", "path": str(root), "role": "singing", "sample_rate": SAMPLE_RATE,
        "config_path": str(root / BANK / "dsconfig.yaml"), "vocoder_path": str(root / VOCODER),
        "source": SOURCE, "revision": "v2.7.0", "attribution": ATTRIBUTION,
        "terms_url": SOURCE + "/blob/main/terms_of_use/Terms_of_Use.zh-CN.md",
        "terms_path": str(root / BANK / "Terms_of_Use.zh-CN.pdf"), "allow_output_training": False,
        "vocoder_license": "CC BY-NC-SA 4.0", "vocoder_source": "https://github.com/openvpi/vocoders"}


def model_readiness(model: dict) -> dict:
    root = Path(model["path"])
    required = [root / BANK / "dsconfig.yaml", root / VOCODER,
        root / BANK / "0101_qixuan_muon1_acoustic.qixuan.onnx",
        root / BANK / "0101_qixuan_muon1_acoustic.phonemes.json", root / BANK / "dsdur" / "dsdict-zh.yaml"]
    for sub, kind in (("dsdur", "dur"), ("dspitch", "pitch"), ("dsvariance", "variance")):
        folder = root / BANK / sub
        if not folder.is_dir() or not list(folder.glob(f"*.{kind}.onnx")) or not list(folder.glob("*.linguistic.onnx")):
            required.append(folder / f"{kind} 与 linguistic 模型")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    return {"ready": not missing, "missing": missing}


def download_assets(root: Path, log: Callable[[str], None] = print) -> dict:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for scheme, proxy in getproxies().items():
        if scheme in {"http", "https"}:
            os.environ.setdefault(scheme.upper() + "_PROXY", proxy)
    for asset in ASSETS:
        target = root / asset["name"]

        def verified(path: Path) -> bool:
            if not path.is_file() or path.stat().st_size != asset["size"]:
                return False
            with path.open("rb") as source:
                return hashlib.file_digest(source, "sha256").hexdigest() == asset["sha256"]

        if not verified(target):
            partial = target.with_name(target.name + ".download")
            if partial.exists() and partial.stat().st_size >= asset["size"] and not verified(partial):
                partial.unlink()
            offset = partial.stat().st_size if partial.exists() else 0
            while offset < asset["size"]:
                end = min(offset + 16 * 1024 * 1024, asset["size"]) - 1
                for attempt in range(3):
                    try:
                        request = Request(asset["url"], headers={"Range": f"bytes={offset}-{end}"})
                        with urlopen(request, timeout=60) as response:
                            if response.status == 200:
                                # Some official release caches return the entire archive.
                                with partial.open("wb") as output:
                                    while block := response.read(1024 * 1024):
                                        output.write(block)
                                offset = partial.stat().st_size
                                break
                            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                                raise RuntimeError("声库下载服务器返回了错误的范围")
                            data = response.read(end - offset + 2)
                        if len(data) != end - offset + 1:
                            raise RuntimeError("声库下载范围不完整")
                        with partial.open("ab") as output:
                            output.write(data)
                        offset += len(data)
                        break
                    except (OSError, RuntimeError):
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
                log(f"{asset['name']}：{offset / 1048576:.1f} / {asset['size'] / 1048576:.1f} MiB")
            if not verified(partial):
                raise RuntimeError("声库 SHA256 校验失败，请重新下载")
            partial.replace(target)
        log("官方发布文件 SHA256 校验通过：" + asset["name"])
        with zipfile.ZipFile(target) as archive:
            for member in archive.infolist():
                destination = (root / member.filename).resolve()
                if not destination.is_relative_to(root) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("声库压缩包含不安全的路径")
            archive.extractall(root)
    result = model_metadata(root)
    if not model_readiness(result)["ready"]:
        raise RuntimeError("声库文件不完整，无法进行本地演唱")
    from audio_studio.store import write_json
    write_json(root / "download-verification.json", {"assets": ASSETS, "source": SOURCE, "model_version": "2.7.0"})
    return result


def _configuration(folder: Path) -> dict:
    import yaml
    return yaml.load((folder / "dsconfig.yaml").read_text(encoding="utf-8"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


@lru_cache(maxsize=4)
def _dictionary(source: str, modified: int) -> dict:
    """The author's 14MB dictionary is immutable between file revisions."""
    import yaml
    entries = yaml.load(Path(source).read_text(encoding="utf-8"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["entries"]
    return {entry["grapheme"]: entry["phonemes"] for entry in entries}


def _load_dictionary(path: Path) -> dict:
    path = path.resolve()
    return _dictionary(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _symbol_types(source: str, modified: int) -> dict[str, str]:
    """Read the small symbol header without reparsing the 14MB entries list."""
    import yaml
    header = []
    with Path(source).open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("entries:"):
                break
            header.append(line)
    data = yaml.load("".join(header), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    return {symbol["symbol"]: symbol["type"] for symbol in data["symbols"]}


def _phonemes(lyric: str, dictionary: dict) -> list[list[str]]:
    from pypinyin import Style, lazy_pinyin
    from audio_studio.score import lyric_units
    units = lyric_units(lyric)
    syllables = []
    for unit in units:
        if unit == "-":
            raise ValueError("延音符号 - 须单独用于后续音符")
        if re_chinese(unit):
            syllables.extend(lazy_pinyin(unit, style=Style.NORMAL))
        else:
            syllables.append(unit.lower().rstrip("12345").replace("ü", "v"))
    result = []
    for syllable in syllables:
        if syllable not in dictionary:
            raise ValueError(f"声库词典不支持读音 {syllable}，请用普通话汉字或拼音校正歌词")
        result.append(dictionary[syllable])
    if not result:
        raise ValueError("演唱音符需要普通话汉字或拼音歌词")
    return result


def re_chinese(text: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in text)


def _contextual_notes(notes: list[dict]) -> list[dict]:
    """Romanize Chinese across note boundaries without overriding pinyin hints."""
    from pypinyin import Style, lazy_pinyin
    from audio_studio.score import lyric_units
    result = [dict(note) for note in notes]
    pending = []

    def flush() -> None:
        if not pending:
            return
        syllables = lazy_pinyin("".join(text for _, text in pending), style=Style.NORMAL)
        cursor = 0
        for index, text in pending:
            result[index]["lyric"] = " ".join(syllables[cursor:cursor + len(text)])
            cursor += len(text)
        pending.clear()

    for index, note in enumerate(result):
        if note["midi"] is None:
            flush()
            continue
        if note["lyric"] == "-":
            continue
        units = lyric_units(note["lyric"])
        if units and all(re_chinese(unit) for unit in units):
            pending.append((index, "".join(units)))
        else:
            flush()
    flush()
    return result


def validate_score(model: dict, score: dict) -> dict:
    import yaml
    validated = SingingScore.model_validate(score)
    dictionary_path = Path(model["path"]) / BANK / "dsdur" / "dsdict-zh.yaml"
    if not dictionary_path.is_file():
        raise ValueError("请先下载完整声库，再校验歌词读音")
    dictionary = _load_dictionary(dictionary_path)
    for note in _contextual_notes(validated.model_dump()["notes"]):
        if note["midi"] is not None and note["lyric"] != "-":
            phones = _phonemes(note["lyric"], dictionary)
            if sum(len(unit) for unit in phones) > round(note["duration"] * SAMPLE_RATE / HOP):
                raise ValueError("歌词音素多于该音符的可用时间，请拆分歌词或延长音符")
    return {"valid": True, "notes": len(validated.notes), "duration_seconds": validated.notes[-1].start + validated.notes[-1].duration,
        "title": validated.title, "attribution": ATTRIBUTION}


def _phrase_inputs(notes: list[dict], dictionary: dict):
    import numpy as np
    rate = SAMPLE_RATE / HOP
    start = notes[0]["start"]
    inner = round((notes[-1]["start"] + notes[-1]["duration"] - start) * rate)
    pad = PHRASE_MARGIN_FRAMES
    total = inner + pad * 2
    words = [{"phones": ["SP"], "frames": pad, "midi": 60}]
    midi_values = np.full(total, 60, dtype=np.float32)
    voiced = np.zeros(total, dtype=bool)
    note_values, note_durations, note_rests = [60.], [pad], [True]
    previous = 0
    for note in notes:
        left = round((note["start"] - start) * rate)
        right = round((note["start"] + note["duration"] - start) * rate)
        frames = right - left
        if left > previous:
            words.append({"phones": ["SP"], "frames": left - previous, "midi": 60})
            note_values.append(60.); note_durations.append(left - previous); note_rests.append(True)
        pitch = note["midi"]
        note_values.append(float(pitch or 60)); note_durations.append(frames); note_rests.append(pitch is None)
        if pitch is None:
            words.append({"phones": ["SP"], "frames": frames, "midi": 60})
        elif note["lyric"] == "-":
            if words[-1]["phones"] == ["SP"]:
                raise ValueError("延音符号 - 前面须为连续的演唱音符")
            words[-1]["frames"] += frames
        else:
            syllables = _phonemes(note["lyric"], dictionary)
            boundaries = np.rint(np.linspace(0, frames, len(syllables) + 1)).astype(int)
            for index, phones in enumerate(syllables):
                words.append({"phones": phones, "frames": int(boundaries[index + 1] - boundaries[index]), "midi": pitch})
        if pitch is not None:
            midi_values[pad + left:pad + right] = pitch
            voiced[pad + left:pad + right] = True
        previous = right
    words.append({"phones": ["SP"], "frames": pad, "midi": 60})
    note_values.append(60.); note_durations.append(pad); note_rests.append(True)
    return words, midi_values, voiced, note_values, note_durations, note_rests, pad, inner


def _vowel_aligned_groups(words: list[dict], symbol_types: dict[str, str]) -> list[dict]:
    """Align vowels to notes, assigning the following onset to the prior span.

    Matches OpenUtau DiffSingerBasePhonemizer ProcessWord/alignment at commit
    a9547895f93e23371949a12d34e98deb5dad0bda. For ni/hao this produces
    [SP,n], [i,h], [ao], rather than delaying both vowels inside their notes.
    """
    groups = []
    for word in words:
        phones = list(word["phones"])
        anchors = [index for index, phone in enumerate(phones) if symbol_types.get(phone) == "vowel"]
        anchor = anchors[0] if anchors else 0
        if anchor > 1 and symbol_types.get(phones[anchor - 1]) == "glide":
            anchor -= 1
        if anchor:
            if not groups:
                raise ValueError("歌声发音缺少起始辅音的前置时间")
            if len(groups) > 1 and groups[-1]["phones"] == ["SP"] and groups[-1]["frames"] <= anchor:
                gap = groups.pop()
                groups[-1]["phones"].extend(gap["phones"])
                groups[-1]["frames"] += gap["frames"]
            groups[-1]["phones"].extend(phones[:anchor])
        groups.append({**word, "phones": phones[anchor:]})
    non_rest = [word for word in words if any(phone not in {"SP", "AP"} for phone in word["phones"])]
    if non_rest:
        groups[0]["midi"] = non_rest[0]["midi"]
        groups[-1]["midi"] = non_rest[-1]["midi"]
    # Upstream duration inference gives initial consonants 500ms of conditioning
    # space, then retains their raw predicted duration before the vowel anchor.
    groups[0]["condition_frames"] = round(.5 * SAMPLE_RATE / HOP)
    return groups


def _aligned_phone_durations(groups: list[dict], prediction):
    import numpy as np
    durations = []
    cursor = 0
    for index, group in enumerate(groups):
        count, length = len(group["phones"]), int(group["frames"])
        if length < count:
            raise ValueError("音符或间隔过短，无法容纳歌词音素")
        weights = np.maximum(np.asarray(prediction[cursor:cursor + count]), .001)
        if len(weights) != count:
            raise RuntimeError("音素时长预测数量与歌词不一致")
        if index == 0 and group["phones"][0] == "SP" and count > 1:
            consonants = np.maximum(1, np.rint(weights[1:])).astype(int)
            if int(consonants.sum()) <= length - 1:
                values = np.concatenate(([length - int(consonants.sum())], consonants))
            else:
                edges = np.rint(np.concatenate(([0.], np.cumsum(weights) / weights.sum() * (length - count)))).astype(int)
                values = np.diff(edges) + 1
        else:
            edges = np.rint(np.concatenate(([0.], np.cumsum(weights) / weights.sum() * (length - count)))).astype(int)
            values = np.diff(edges) + 1
        durations.extend(values)
        cursor += count
    return np.asarray([durations], dtype=np.int64)


def _rest_filled_note_pitches(values: list[float], rests: list[bool]):
    """Use adjacent real notes for rest conditioning, as OpenUtau does."""
    import numpy as np
    result = np.asarray(values, dtype=np.float32).copy()
    if all(rests):
        result.fill(60)
        return result
    index = 0
    while index < len(rests):
        if not rests[index]:
            index += 1
            continue
        end = index + 1
        while end < len(rests) and rests[end]:
            end += 1
        if index == 0:
            result[index:end] = result[end]
        elif end == len(rests):
            result[index:end] = result[index - 1]
        else:
            middle = (index + end + 1) // 2
            result[index:middle] = result[index - 1]
            result[middle:end] = result[end]
        index = end
    return result


def _continuous_pitch(pitch, voiced):
    """Fill padding/rest regions rather than injecting artificial F0=0 jumps."""
    import numpy as np
    values = np.asarray(pitch, dtype=np.float32)
    mask = np.asarray(voiced, dtype=bool)
    if values.shape != (1, len(mask)) or not np.isfinite(values).all():
        raise RuntimeError("歌声音高曲线格式或数值无效")
    anchors = np.flatnonzero(mask)
    if not len(anchors):
        return np.full_like(values, 60)
    return np.interp(np.arange(len(mask)), anchors, values[0, anchors])[None].astype(np.float32)


def _pitch_inputs(encoder, durations, baseline, note_values, note_durations, note_rests, steps: int, expressiveness: float) -> dict:
    import numpy as np
    return {"encoder_out": encoder, "ph_dur": durations,
        "note_midi": _rest_filled_note_pitches(note_values, note_rests)[None],
        "note_rest": np.asarray([note_rests], dtype=bool), "note_dur": np.asarray([note_durations], dtype=np.int64),
        "pitch": baseline, "expr": np.full_like(baseline, expressiveness),
        "retake": np.ones_like(baseline, dtype=bool), "steps": np.asarray(steps, dtype=np.int64)}


def _acoustic_f0(pitch, voiced):
    import numpy as np
    continuous = _continuous_pitch(pitch, voiced)
    values = (440 * np.exp2((continuous - 69) / 12)).astype(np.float32)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise RuntimeError("歌声模型返回了无效的连续 F0")
    return values


def _acoustic_variances(breathiness, voicing):
    import numpy as np
    return tuple(np.clip(value, -96, 0).astype(np.float32) for value in (breathiness, voicing))


def _place_phrase(output, wave, left: int, occupied_until: int) -> int:
    """Retain phrase margins and crossfade overlapping boundaries once."""
    import numpy as np
    right = min(len(output), left + len(wave))
    segment = wave[:right - left]
    overlap = min(max(0, occupied_until - left), len(segment))
    if overlap:
        fade = np.linspace(0, 1, overlap, dtype=np.float32)
        output[left:left + overlap] = output[left:left + overlap] * (1 - fade) + segment[:overlap] * fade
    output[left + overlap:right] = segment[overlap:]
    return max(occupied_until, right)


def synthesize(model: dict, score: dict, output_path: Path, steps: int = 20, expressiveness: float = .7,
               log: Callable[[str], None] = print) -> dict:
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    import yaml
    readiness = model_readiness(model)
    if not readiness["ready"]:
        raise ValueError("DiffSinger 声库缺少文件：" + ", ".join(readiness["missing"]))
    validate_score(model, score)
    validated = SingingScore.model_validate(score).model_dump()
    if not 4 <= steps <= 100 or not math.isfinite(expressiveness) or not 0 <= expressiveness <= 1:
        raise ValueError("采样步数或表现力参数无效")
    root = Path(model["path"])
    bank = root / BANK
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    sessions, configs = {}, {}
    for key, folder, names in (("dur", bank / "dsdur", ("linguistic", "dur")),
                               ("pitch", bank / "dspitch", ("linguistic", "pitch")),
                               ("variance", bank / "dsvariance", ("linguistic", "variance")),
                               ("acoustic", bank, ("acoustic",))):
        config = _configuration(folder)
        config["_folder"] = folder
        config["_phones"] = json.loads((folder / config["phonemes"]).read_text(encoding="utf-8"))
        configs[key] = config
        for name in names:
            sessions[key + "_" + name] = ort.InferenceSession(str(folder / config[name]), sess_options=options, providers=["CPUExecutionProvider"])
    vocoder = ort.InferenceSession(str(root / VOCODER), sess_options=options, providers=["CPUExecutionProvider"])
    dictionary_path = bank / "dsdur" / "dsdict-zh.yaml"
    dictionary = _load_dictionary(dictionary_path)
    symbol_types = _symbol_types(str(dictionary_path.resolve()), dictionary_path.stat().st_mtime_ns)
    phrases = []
    for note in _contextual_notes(validated["notes"]):
        if not phrases or (note["lyric"] != "-" and (note["start"] + note["duration"] - phrases[-1][0]["start"] > 12
                or note["start"] - (phrases[-1][-1]["start"] + phrases[-1][-1]["duration"]) > .3)):
            phrases.append([])
        phrases[-1].append(note)
    total = round((validated["notes"][-1]["start"] + validated["notes"][-1]["duration"]) * SAMPLE_RATE / HOP) * HOP
    output = np.zeros(total + 2 * PHRASE_MARGIN_FRAMES * HOP, dtype=np.float32)
    occupied_until = 0

    def token_arrays(kind, phones):
        mapping = configs[kind]["_phones"]
        try:
            tokens = [mapping[phone] for phone in phones]
        except KeyError as exc:
            raise ValueError(f"声库音素表不支持 {exc.args[0]}") from exc
        languages = [3 if phone.startswith("zh/") else 0 for phone in phones]
        return {"tokens": np.asarray([tokens], dtype=np.int64), "languages": np.asarray([languages], dtype=np.int64)}

    def run(name, inputs):
        values = sessions[name].run(None, inputs)
        if any(not np.isfinite(value).all() for value in values):
            raise RuntimeError("歌声模型返回了非有限数值：" + name)
        return values

    for index, notes in enumerate(phrases):
        log(f"DiffSinger 乐句 {index + 1}/{len(phrases)}：{notes[0]['start']:.2f} 秒起，预测发音与演唱参数。")
        words, midi_values, voiced, note_values, note_durations, note_rests, pad, inner = _phrase_inputs(notes, dictionary)
        words = _vowel_aligned_groups(words, symbol_types)
        phones = [phone for word in words for phone in word["phones"]]
        divisions = np.asarray([[len(word["phones"]) for word in words]], dtype=np.int64)
        word_durations = np.asarray([[word.get("condition_frames", word["frames"]) for word in words]], dtype=np.int64)
        encoder_out, masks = run("dur_linguistic", {**token_arrays("dur", phones), "word_div": divisions, "word_dur": word_durations})
        phone_midi = np.asarray([[word["midi"] for word in words for _ in word["phones"]]], dtype=np.int64)
        prediction = run("dur_dur", {"encoder_out": encoder_out, "x_masks": masks, "ph_midi": phone_midi})[0][0]
        durations = _aligned_phone_durations(words, prediction)
        assert int(durations.sum()) == len(midi_values)
        pitch_encoder, _ = run("pitch_linguistic", {**token_arrays("pitch", phones), "ph_dur": durations})
        baseline = midi_values[None]
        pitch = run("pitch_pitch", _pitch_inputs(pitch_encoder, durations, baseline, note_values,
            note_durations, note_rests, steps, expressiveness))[0]
        pitch = _continuous_pitch(pitch, voiced)
        variance_encoder, _ = run("variance_linguistic", {**token_arrays("variance", phones), "ph_dur": durations})
        breathiness, voicing = run("variance_variance", {"encoder_out": variance_encoder, "ph_dur": durations,
            "pitch": pitch, "breathiness": np.zeros_like(pitch), "voicing": np.zeros_like(pitch),
            "retake": np.ones((1, pitch.shape[1], 2), dtype=bool), "steps": np.asarray(steps, dtype=np.int64)})
        breathiness, voicing = _acoustic_variances(breathiness, voicing)
        f0 = _acoustic_f0(pitch, voiced)
        log(f"DiffSinger 乐句 {index + 1}/{len(phrases)}：真实声学模型采样 {steps} 步并解码音频。")
        mel = run("acoustic_acoustic", {**token_arrays("acoustic", phones), "durations": durations, "f0": f0,
            "breathiness": breathiness, "voicing": voicing, "gender": np.zeros_like(f0), "velocity": np.ones_like(f0),
            "depth": np.asarray(.6, dtype=np.float32), "steps": np.asarray(steps, dtype=np.int64)})[0]
        wave = vocoder.run(None, {"mel": mel, "f0": f0})[0].reshape(-1)
        if not np.isfinite(wave).all() or len(wave) < (2 * pad + inner) * HOP:
            raise RuntimeError("神经声码器未返回完整有效的歌声音频")
        wave = wave[:(2 * pad + inner) * HOP]
        left = round(notes[0]["start"] * SAMPLE_RATE / HOP) * HOP
        occupied_until = _place_phrase(output, wave, left, occupied_until)
    peak = float(np.max(np.abs(output), initial=0))
    if peak < 1e-6:
        raise RuntimeError("歌声模型输出只有静音，未保存音频")
    if peak > .98:
        output *= .98 / peak
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, output, SAMPLE_RATE, subtype="PCM_16")
    return {"path": str(output_path), "sample_rate": SAMPLE_RATE, "duration_seconds": len(output) / SAMPLE_RATE,
        "engine": "diffsinger", "device": "cpu", "phrases": len(phrases), "notes": len(validated["notes"]),
        "steps": steps, "attribution": ATTRIBUTION, "allow_output_training": False,
        "quality_revision": QUALITY_REVISION, "lead_in_seconds": PHRASE_MARGIN_FRAMES * HOP / SAMPLE_RATE,
        "tail_seconds": PHRASE_MARGIN_FRAMES * HOP / SAMPLE_RATE,
        "musical_duration_seconds": total / SAMPLE_RATE}
