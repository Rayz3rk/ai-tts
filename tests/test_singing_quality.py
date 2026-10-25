"""Numerical regressions for the audited OpenUtau inference contract."""

import numpy as np
import pytest

from audio_studio.singing import (
    HOP, SAMPLE_RATE, PHRASE_MARGIN_FRAMES, _acoustic_f0, _acoustic_variances,
    _aligned_phone_durations, _contextual_notes, _continuous_pitch, _phrase_inputs,
    _pitch_inputs, _place_phrase, _rest_filled_note_pitches, _vowel_aligned_groups,
)


DICTIONARY = {"ni": ["zh/n", "zh/i"], "hao": ["zh/h", "zh/ao"]}
TYPES = {"SP": "vowel", "zh/n": "liquid", "zh/i": "vowel", "zh/h": "fricative", "zh/ao": "vowel"}


def note(lyric, midi=60, start=0, duration=.5):
    return {"lyric": lyric, "midi": midi, "start": start, "duration": duration}


def test_vowels_start_on_note_anchors_and_consonants_precede_them():
    notes = [note("ni"), note("hao", 62, .5)]
    words, pitches, _, _, _, _, pad, _ = _phrase_inputs(notes, DICTIONARY)
    groups = _vowel_aligned_groups(words, TYPES)
    assert [group["phones"] for group in groups] == [["SP", "zh/n"], ["zh/i", "zh/h"], ["zh/ao"], ["SP"]]
    # The h preceding hao belongs to the preceding interval and uses its tone.
    assert groups[1]["midi"] == 60 and groups[2]["midi"] == 62
    durations = _aligned_phone_durations(groups, [20, 4, 32, 5, 30, 13])[0]
    assert durations[1] == 4  # initial n keeps its predicted length
    assert durations[:2].sum() == pad
    assert durations[:4].sum() == pad + round(.5 * SAMPLE_RATE / HOP)
    assert durations.sum() == len(pitches)
    assert durations[0] >= 1 and np.all(durations >= 1)


def test_melisma_and_real_rests_keep_complete_frame_timeline():
    notes = [note("ni"), note("-", 62, .5), note("", None, 1, .3), note("hao", 64, 1.3)]
    words, pitches, voiced, _, _, _, pad, inner = _phrase_inputs(notes, DICTIONARY)
    groups = _vowel_aligned_groups(words, TYPES)
    assert groups[1]["phones"] == ["zh/i"]
    assert groups[1]["frames"] == round(SAMPLE_RATE / HOP)
    assert groups[2]["phones"] == ["SP", "zh/h"]
    count = sum(len(group["phones"]) for group in groups)
    durations = _aligned_phone_durations(groups, np.arange(1, count + 1, dtype=float))[0]
    assert durations.sum() == sum(group["frames"] for group in groups) == inner + 2 * pad
    assert not voiced[pad + round(1.1 * SAMPLE_RATE / HOP)]


def test_short_midi_gaps_do_not_make_onset_groups_impossible():
    notes = [note("ni"), note("hao", 62, .512)]
    words, pitches, *_ = _phrase_inputs(notes, DICTIONARY)
    groups = _vowel_aligned_groups(words, TYPES)
    prediction = np.ones(sum(len(group["phones"]) for group in groups))
    assert _aligned_phone_durations(groups, prediction).sum() == len(pitches)


def test_pitch_and_acoustic_f0_remain_continuous_across_padding_and_rest():
    pitch = np.array([[20, 60, 62, 99, 99, 64, 65, 90]], dtype=np.float32)
    voiced = np.array([False, True, True, False, False, True, True, False])
    continuous = _continuous_pitch(pitch, voiced)
    assert continuous[0, 0] == 60 and continuous[0, -1] == 65
    assert 62 < continuous[0, 3] < continuous[0, 4] < 64
    f0 = _acoustic_f0(pitch, voiced)
    assert np.isfinite(f0).all() and np.all(f0 > 0)
    assert f0[0, 0] == pytest.approx(261.6256, rel=1e-5)


def test_rest_note_pitch_uses_nearest_real_notes_and_not_constant_c4():
    filled = _rest_filled_note_pitches([60, 65, 60, 60, 72, 60], [True, False, True, True, False, True])
    assert filled.tolist() == [65, 65, 65, 72, 72, 72]


def test_expressiveness_is_a_model_input_and_sampling_scalar_rank_is_retained():
    baseline = np.full((1, 20), 60, dtype=np.float32)
    inputs = _pitch_inputs(np.zeros((1, 3, 256)), np.array([[2, 16, 2]]), baseline,
        [60, 65, 60], [2, 16, 2], [True, False, True], 32, .35)
    assert np.allclose(inputs["expr"], .35)
    assert inputs["steps"].shape == ()
    assert inputs["note_midi"].tolist() == [[65, 65, 65]]
    assert inputs["retake"].all()


def test_acoustic_voice_parameters_are_in_upstream_db_range():
    breath, voice = _acoustic_variances(np.array([[-110., -12., 3.]]), np.array([[2., -6., -100.]]))
    assert breath.tolist() == [[-96, -12, 0]]
    assert voice.tolist() == [[0, -6, -96]]
    assert breath.dtype == voice.dtype == np.float32


def test_phrase_margins_and_crossfade_preserve_release_without_double_amplitude():
    pad = PHRASE_MARGIN_FRAMES * HOP
    wave = np.ones(4 * pad, dtype=np.float32) * .5
    output = np.zeros(7 * pad, dtype=np.float32)
    end = _place_phrase(output, wave, 0, 0)
    end = _place_phrase(output, wave, 3 * pad, end)
    assert end == len(output)
    assert np.allclose(output, .5)


def test_chinese_polyphones_use_context_across_notes_and_preserve_pinyin_overrides():
    notes = [note("音"), note("乐", start=.5), note("-", start=1), note("带", start=1.5), note("来", start=2)]
    contextual = _contextual_notes(notes)
    assert [entry["lyric"] for entry in contextual] == ["yin", "yue", "-", "dai", "lai"]
    assert notes[1]["lyric"] == "乐"  # user score is never mutated
    explicit = _contextual_notes([note("音"), note("le4", start=.5), note("-", start=1)])
    assert [entry["lyric"] for entry in explicit] == ["yin", "le4", "-"]
