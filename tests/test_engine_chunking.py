"""Long text must reach the model without Kokoro's silent phoneme truncation."""

import pytest

from audio_studio.engines import _phoneme_segments


def test_long_chinese_with_expanding_phonemes_preserves_every_character():
    text = "中华人民共和国的每个汉字都需要完整朗读" * 60
    chunks = _phoneme_segments(text, lambda source: ("ʈʂʰ˥˩" * (len(source) * 4), None))
    assert "".join(source for source, _ in chunks) == text
    assert len(chunks) > 1
    assert all(0 < len(phonemes) <= 510 for _, phonemes in chunks)


def test_unrepresentable_single_character_fails_instead_of_truncating():
    with pytest.raises(ValueError, match="音素长度"):
        _phoneme_segments("字", lambda source: ("a" * 511, None))
