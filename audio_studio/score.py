"""Monophonic MIDI timing and explicit lyric-to-note alignment."""

from bisect import bisect_right
import io
import re


def lyric_units(text: str) -> list[str]:
    """Mandarin characters, spaced pinyin and explicit melisma markers."""
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-züÜvV]+[1-5]?|-", text)


def score_from_text(title: str, lyrics: str, melody: str, bpm: float = 120) -> dict:
    """Convert a complete pasted melody to seconds without per-note data entry.

    Numeric 1..7 uses C major, note names such as F#4 are accepted, :N gives
    beats, 0/rest gives a rest, - holds, and ~ continues the preceding syllable
    on a new pitch. This imports an explicit melody; it does not compose one.
    """
    import math
    from audio_studio.schemas import SingingScore
    if not math.isfinite(bpm) or not 40 <= bpm <= 240:
        raise ValueError("速度须为 40–240 BPM")
    tokens = re.sub(r"[|,，;；]", " ", melody).split()
    if not tokens or len(tokens) > 2000:
        raise ValueError("请粘贴 1–2000 个旋律音符，或导入 MIDI")
    syllables = lyric_units(lyrics)
    notes, cursor, lyric_index = [], 0.0, 0
    previous_pitch = None
    scale = [60, 62, 64, 65, 67, 69, 71]
    for token in tokens:
        parts = token.split(":")
        if len(parts) > 2:
            raise ValueError(f"旋律写法无效：{token}，可用 3:2 或 E4:2 表示两拍")
        symbol = parts[0]
        try:
            beats = float(parts[1]) if len(parts) == 2 else 1.0
        except ValueError:
            raise ValueError(f"音符拍数无效：{token}") from None
        seconds = beats * 60 / bpm
        if not math.isfinite(seconds) or not .08 <= seconds <= 20:
            raise ValueError(f"音符 {token} 的时长须为 0.08–20 秒，请调整 BPM 或拍数")
        continued = symbol.endswith("~") or symbol == "-"
        symbol = symbol.rstrip("~")
        if symbol == "-":
            pitch = previous_pitch
        elif symbol.lower() in {"0", "rest", "sp"}:
            pitch = None
        elif re.fullmatch(r"[1-7][+-]?", symbol):
            pitch = scale[int(symbol[0]) - 1] + (12 if symbol.endswith("+") else -12 if symbol.endswith("-") else 0)
        elif symbol.isdigit() and 24 <= int(symbol) <= 96:
            pitch = int(symbol)
        else:
            match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", symbol)
            if not match:
                raise ValueError(f"无法识别音高 {symbol}，请用简谱 1–7、C4 等音名或 0 休止")
            pitch = (int(match[3]) + 1) * 12 + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[match[1].upper()]
            pitch += 1 if match[2] == "#" else -1 if match[2] == "b" else 0
        if continued:
            if previous_pitch is None or pitch is None:
                raise ValueError("延音或一字多音须紧接演唱音符，不能放在开头或休止之后")
            lyric = "-"
        elif pitch is None:
            lyric = "SP"
        else:
            if lyric_index >= len(syllables):
                raise ValueError("旋律的填词位置多于歌词；一字多音的后续音符加 ~，同音延长可用 -")
            lyric = syllables[lyric_index]
            lyric_index += 1
        notes.append({"lyric": lyric, "midi": pitch, "start": round(cursor, 6), "duration": round(seconds, 6)})
        cursor = round(cursor + seconds, 6)
        previous_pitch = pitch
    if lyric_index != len(syllables):
        raise ValueError(f"歌词有 {len(syllables)} 个字/音节，旋律只有 {lyric_index} 个填词位置，请补齐旋律或分段导入")
    score = SingingScore.model_validate({"title": title, "notes": notes}).model_dump()
    warnings = []
    average = sum(note["duration"] for note in notes if note["midi"] is not None) / max(1, lyric_index)
    if average > 1.2:
        warnings.append("平均每字超过 1.2 秒；若不希望长音，请提高 BPM 或减少音符拍数")
    return {"score": score, "duration_seconds": cursor, "syllables": lyric_index,
            "average_syllable_seconds": average, "warnings": warnings}


def import_midi(data: bytes, lyrics: str = "", track_index: int | None = None) -> dict:
    import mido
    try:
        midi = mido.MidiFile(file=io.BytesIO(data))
    except (ValueError, OSError, EOFError, KeyError) as exc:
        raise ValueError("无法读取标准 MIDI 文件") from exc
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("仅支持共享时间轴的 MIDI type 0/1 与 PPQ 时间格式")
    events, tempos = [], {0: 500000}
    for index, track in enumerate(midi.tracks):
        tick = 0
        track_events = []
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                if message.tempo <= 0:
                    raise ValueError("MIDI 速度值无效")
                tempos[tick] = message.tempo
            track_events.append((tick, message))
        events.append(track_events)
    positions, elapsed, tempo_values = [], [], []
    previous_tick, seconds, previous_tempo = 0, 0.0, 500000
    for tick, tempo in sorted(tempos.items()):
        seconds += mido.tick2second(tick - previous_tick, midi.ticks_per_beat, previous_tempo)
        positions.append(tick); elapsed.append(seconds); tempo_values.append(tempo)
        previous_tick, previous_tempo = tick, tempo

    def at(tick: int) -> float:
        segment = bisect_right(positions, tick) - 1
        return elapsed[segment] + mido.tick2second(tick - positions[segment], midi.ticks_per_beat, tempo_values[segment])

    tracks = []
    for index, track_events in enumerate(events):
        active, notes, warnings = {}, [], []
        name = next((message.name for _, message in track_events if message.type == "track_name"), f"轨道 {index + 1}")
        for tick, message in track_events:
            if message.type not in {"note_on", "note_off"} or message.channel == 9:
                continue
            key = (message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                if key in active:
                    warnings.append("同音符存在重复按下事件，请检查原 MIDI")
                active[key] = tick
            elif key in active:
                start = at(active.pop(key)); duration = at(tick) - start
                if duration > 0:
                    notes.append({"lyric": "", "midi": message.note, "start": round(start, 6), "duration": round(duration, 6)})
        if active:
            warnings.append("有音符缺少结束事件，未导入这些音符")
        if not notes:
            continue
        notes.sort(key=lambda note: (note["start"], note["midi"]))
        monophonic = all(notes[n]["start"] >= notes[n-1]["start"] + notes[n-1]["duration"] - 1e-5 for n in range(1, len(notes)))
        tracks.append({"index": index, "name": name, "notes": notes, "note_count": len(notes), "monophonic": monophonic, "warnings": warnings})
    if not tracks:
        raise ValueError("MIDI 中没有可导入的旋律音符（打击乐轨道不作旋律）")
    chosen = next((track for track in tracks if track["index"] == track_index), None)
    if track_index is None and len(tracks) == 1:
        chosen = tracks[0]
    errors = []
    if chosen:
        units = lyric_units(lyrics)
        if units and len(units) != len(chosen["notes"]):
            errors.append(f"歌词含 {len(units)} 个字/音节，旋律有 {len(chosen['notes'])} 个音符；一字多音请用 - 占位")
        elif units:
            for note, unit in zip(chosen["notes"], units):
                note["lyric"] = unit
        else:
            errors.append("请填写歌词并导入，或在音符表中填写每个音符的歌词")
        if not chosen["monophonic"]:
            errors.append("所选轨道包含重叠音符，请选择单声部旋律轨道")
    elif track_index is not None:
        raise ValueError("选择的 MIDI 旋律轨道不存在")
    return {"tracks": tracks, "selected_track": chosen["index"] if chosen else None,
        "notes": chosen["notes"] if chosen else [], "errors": errors,
        "requires_track_selection": chosen is None, "tempo_changes": len(tempos)}
