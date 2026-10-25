"""Score correctness and MIDI timing contracts; neural inference is separate."""

import io
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from audio_studio.api import create_app
from audio_studio.schemas import SingingScore
from audio_studio.score import import_midi, lyric_units, score_from_text
from audio_studio.singing import _phrase_inputs, model_metadata, model_readiness


def midi_bytes(*, second_track: bool = False, overlap: bool = False) -> bytes:
    mido = pytest.importorskip("mido")
    midi = mido.MidiFile(ticks_per_beat=480)
    tempo = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000), mido.MetaMessage("set_tempo", tempo=400000, time=480)])
    track = mido.MidiTrack([mido.MetaMessage("track_name", name="Melody"), mido.Message("note_on", note=60, velocity=80),
        mido.Message("note_off", note=60, time=480), mido.Message("note_on", note=62, velocity=80), mido.Message("note_off", note=62, time=480)])
    midi.tracks.extend([tempo, track])
    if second_track:
        midi.tracks.append(mido.MidiTrack([mido.Message("note_on", note=64), mido.Message("note_off", note=64, time=960)]))
    if overlap:
        midi.tracks[1] = mido.MidiTrack([mido.Message("note_on", note=60), mido.Message("note_on", note=64),
            mido.Message("note_off", note=60, time=480), mido.Message("note_off", note=64)])
    data = io.BytesIO(); midi.save(file=data)
    return data.getvalue()


def test_midi_global_tempo_map_and_lyric_alignment():
    result = import_midi(midi_bytes(), "你好")
    assert result["selected_track"] == 1 and result["tempo_changes"] == 2
    assert not result["errors"]
    assert [note["duration"] for note in result["notes"]] == pytest.approx([.5, .4])
    assert result["notes"][1]["start"] == .5
    assert [note["lyric"] for note in result["notes"]] == ["你", "好"]


def test_bulk_lyrics_and_melody_use_bpm_and_handle_rests_and_melisma():
    report = score_from_text("歌曲", "你好世界", "1 0 2 3~ 3 4:2", 120)
    notes = report["score"]["notes"]
    assert report["syllables"] == 4 and report["duration_seconds"] == 3.5
    assert [note["lyric"] for note in notes] == ["你", "SP", "好", "-", "世", "界"]
    assert [note["midi"] for note in notes] == [60, None, 62, 64, 64, 65]
    assert notes[-1]["duration"] == 1
    doubled = score_from_text("歌曲", "你好世界", "C4 D4 E4 F4", 240)
    assert doubled["duration_seconds"] == 1
    with pytest.raises(ValueError, match="歌词"):
        score_from_text("歌曲", "你好世界", "1 2", 120)
    with pytest.raises(ValueError, match="延音"):
        score_from_text("歌曲", "你好", "- 1 2", 120)


def test_midi_requires_explicit_track_selection_and_rejects_polyphony():
    result = import_midi(midi_bytes(second_track=True), "你好")
    assert result["requires_track_selection"] and not result["notes"]
    chosen = import_midi(midi_bytes(second_track=True), "你好", track_index=1)
    assert not chosen["errors"] and len(chosen["notes"]) == 2
    polyphony = import_midi(midi_bytes(overlap=True), "你好")
    assert not polyphony["tracks"][0]["monophonic"] and polyphony["errors"]
    assert import_midi(midi_bytes(), "你好世界")["errors"]
    with pytest.raises(ValueError):
        import_midi(b"invalid MIDI")


@pytest.mark.parametrize("notes", [
    [{"lyric": "你", "midi": 60, "start": 0, "duration": .5}, {"lyric": "好", "midi": 62, "start": .4, "duration": .5}],
    [{"lyric": "-", "midi": 60, "start": 0, "duration": .5}],
    [{"lyric": "", "midi": 60, "start": 0, "duration": .5}],
    [{"lyric": "你", "midi": 160, "start": 0, "duration": .5}],
    [{"lyric": "你", "midi": 60, "start": float("nan"), "duration": .5}],
])
def test_invalid_scores_are_rejected_before_inference(notes):
    with pytest.raises(ValidationError):
        SingingScore.model_validate({"notes": notes})


def test_melisma_rest_and_timeline_frames_are_preserved():
    notes = [{"lyric": "你", "midi": 60, "start": 0, "duration": .5},
        {"lyric": "-", "midi": 62, "start": .5, "duration": .5},
        {"lyric": "好", "midi": 64, "start": 1.2, "duration": .5}]
    words, pitches, voiced, _, durations, _, pad, inner = _phrase_inputs(notes, {"ni": ["zh/n", "zh/i"], "hao": ["zh/h", "zh/ao"]})
    assert words[1]["phones"] == ["zh/n", "zh/i"]
    assert words[1]["frames"] == round(44100 / 512)
    assert sum(word["frames"] for word in words) == len(pitches) == sum(durations)
    assert voiced[pad:pad + inner].sum() > 100 and not voiced[0]
    assert set(np.unique(pitches[voiced])) == {60, 62, 64}
    assert lyric_units("你好，shi4 jie4 -") == ["你", "好", "shi4", "jie4", "-"]


def test_singing_api_keeps_existing_tts_and_exposes_missing_models(tmp_path: Path):
    with TestClient(create_app(tmp_path, database_url="")) as client:
        response = client.get("/api/singing/status")
        assert response.status_code == 200
        assert response.json()["model"]["engine"] == "diffsinger"
        assert not response.json()["model"]["ready"]
        assert len(client.get("/api/presets").json()) == 8
        score = {"notes": [{"lyric": "你", "midi": 60, "start": 0, "duration": .5}]}
        assert client.post("/api/singing/generate", json={"score": score}).status_code == 400
        imported = client.post("/api/singing/midi", data={"lyrics": "你好"}, files={"file": ("melody.mid", midi_bytes(), "audio/midi")})
        assert imported.status_code == 200 and len(imported.json()["notes"]) == 2
        assert not model_readiness(model_metadata(tmp_path))["ready"]
