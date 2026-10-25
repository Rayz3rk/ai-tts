"""Validated, explicit contracts shared by the API and command line."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

IDENTIFIER = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class Preset(StrictModel):
    id: str | None = Field(default=None, pattern=IDENTIFIER)
    name: str = Field(min_length=1, max_length=100)
    category: Literal["narration", "conversation", "news", "audiobook", "assistant", "custom"] = "custom"
    engine: Literal["kokoro", "f5"] = "kokoro"
    model_id: str = Field(default="kokoro-base", pattern=IDENTIFIER)
    language: Literal["a", "b", "z"] = "z"
    voice: str = "zf_xiaobei"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    steps: int = Field(default=32, ge=4, le=64)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    reference_audio: str = ""
    reference_text: str = Field(default="", max_length=3000)

    @model_validator(mode="after")
    def check_voice(self) -> "Preset":
        voices = {"z": {"zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"},
                  "a": {"af_heart", "am_adam"}, "b": {"bf_emma", "bm_george"}}
        if self.engine == "kokoro" and self.voice not in voices[self.language]:
            raise ValueError("音色与语言不匹配；请选择该语言的内置音色")
        if self.engine == "f5" and (not self.reference_audio or not self.reference_text):
            raise ValueError("F5 预设需要参考音频及其准确转写")
        return self


class SynthesisRequest(StrictModel):
    text: str = Field(min_length=1, max_length=5000)
    preset_id: str = Field(pattern=IDENTIFIER)
    speed: float | None = Field(default=None, ge=0.5, le=2)


class DatasetRequest(StrictModel):
    csv_path: str = Field(min_length=1, max_length=4096)
    wavs_dir: str = Field(min_length=1, max_length=4096)


class TrainingRequest(DatasetRequest):
    name: str = Field(default="本地音色", min_length=1, max_length=100)
    model_id: str = Field(default="f5-base", pattern=IDENTIFIER)
    epochs: int = Field(default=10, ge=1, le=10000)
    batch_size: int = Field(default=1, ge=1, le=16)
    learning_rate: float = Field(default=1e-5, ge=1e-7, le=1e-3)
    grad_accumulation_steps: int = Field(default=4, ge=1, le=64)


class RegisterModel(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    checkpoint_path: str = Field(min_length=1, max_length=4096)
    vocab_path: str = Field(min_length=1, max_length=4096)
    vocoder_path: str = Field(default="", max_length=4096)


class AudioAssetUpdate(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    tags: list[str] | None = Field(default=None, max_length=30)
    favorite: bool | None = None

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        tags = list(dict.fromkeys(tag.strip() for tag in value if tag.strip()))
        if any(len(tag) > 80 for tag in tags):
            raise ValueError("每个标签最多 80 字符")
        return tags


class RegisterVoiceModel(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    checkpoint_path: str = Field(min_length=1, max_length=4096)
    index_path: str = Field(default="", max_length=4096)
    speaker_id: int = Field(default=0, ge=0, le=10000)


class CoverRequest(StrictModel):
    model_id: str = Field(pattern=IDENTIFIER)
    source_audio: str = Field(min_length=1, max_length=4096)
    pitch_shift: int = Field(default=0, ge=-24, le=24)
    index_rate: float = Field(default=0.5, ge=0, le=1)
    speaker_id: int = Field(default=0, ge=0, le=10000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    accompaniment_audio: str = Field(default="", max_length=4096)
    vocal_gain: float = Field(default=1, ge=0, le=2)
    accompaniment_gain: float = Field(default=0.7, ge=0, le=2)


class VoiceDatasetRequest(StrictModel):
    wavs_dir: str = Field(min_length=1, max_length=4096)


class VoiceTrainingRequest(VoiceDatasetRequest):
    name: str = Field(default="本地翻唱音色", min_length=1, max_length=100)
    epochs: int = Field(default=20, ge=1, le=10000)
    batch_size: int = Field(default=1, ge=1, le=16)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-3)
    max_steps: int | None = Field(default=None, ge=1, le=1000000)


class SingingNote(StrictModel):
    lyric: str = Field(default="", max_length=40)
    midi: int | None = Field(default=None, ge=24, le=96)
    start: float = Field(ge=0, le=600)
    duration: float = Field(ge=0.08, le=20)


class SingingScore(StrictModel):
    title: str = Field(default="新歌曲", min_length=1, max_length=100)
    notes: list[SingingNote] = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def check_timeline(self) -> "SingingScore":
        previous = 0.0
        voiced = False
        for note in self.notes:
            if note.start < previous - 1e-6:
                raise ValueError("歌声旋律须按时间排序且不能重叠；请选择单声部旋律轨道")
            previous = note.start + note.duration
            if note.midi is not None:
                if not note.lyric:
                    raise ValueError("每个演唱音符需要歌词；一字多音的后续音符用 - 标注")
                if note.lyric == "-" and not voiced:
                    raise ValueError("第一个演唱音符不能是延音符号 -")
                voiced = True
        if not voiced or previous > 600:
            raise ValueError("乐谱须包含演唱音符，且总时长不超过 600 秒")
        return self


class SingingRequest(StrictModel):
    model_id: str = Field(default="diffsinger-qixuan", pattern=IDENTIFIER)
    score: SingingScore
    steps: int = Field(default=20, ge=4, le=100)
    expressiveness: float = Field(default=0.7, ge=0, le=1)
    target_voice_id: str | None = Field(default=None, pattern=IDENTIFIER)
    pitch_shift: int = Field(default=0, ge=-24, le=24)
    index_rate: float = Field(default=0.5, ge=0, le=1)


class BulkScoreRequest(StrictModel):
    title: str = Field(default="新歌曲", min_length=1, max_length=100)
    lyrics: str = Field(min_length=1, max_length=20000)
    melody: str = Field(min_length=1, max_length=20000)
    bpm: float = Field(default=120, ge=40, le=240)


class MusicRequest(StrictModel):
    title: str = Field(default="我的原创歌曲", min_length=1, max_length=100)
    lyrics: str = Field(min_length=1, max_length=4000)
    style: str = Field(default="Mandarin pop, warm vocal, piano and acoustic guitar", min_length=1, max_length=1000)
    duration: float = Field(default=30, ge=10, le=180)
    bpm: int | None = Field(default=None, ge=30, le=300)
    language: Literal["zh", "en", "ja", "ko", "unknown"] = "zh"
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
