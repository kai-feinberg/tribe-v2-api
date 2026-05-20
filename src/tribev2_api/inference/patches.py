from __future__ import annotations

import re
from pathlib import Path

_CURRENT_TEXT: str | None = None


def set_current_text(text: str | None) -> None:
    global _CURRENT_TEXT
    _CURRENT_TEXT = text


def _patched_get_transcript_from_audio(wav_filename, language: str = "english"):
    import pandas as pd
    import soundfile as sf

    try:
        duration = float(sf.info(str(Path(wav_filename))).duration)
    except Exception:
        duration = 30.0

    text = (_CURRENT_TEXT or "audio content placeholder").strip()
    raw_words = text.split()
    if not raw_words:
        return pd.DataFrame(columns=["text", "start", "duration", "sequence_id", "sentence"])

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()] or [text]
    word_duration = max(duration / len(raw_words), 0.01)

    rows = []
    word_index = 0
    for sentence_index, sentence in enumerate(sentences):
        for word in sentence.split():
            if word_index >= len(raw_words):
                break
            rows.append(
                {
                    "text": word.replace('"', ""),
                    "start": word_index * word_duration,
                    "duration": word_duration * 0.9,
                    "sequence_id": sentence_index,
                    "sentence": sentence.replace('"', ""),
                }
            )
            word_index += 1
    return pd.DataFrame(rows)


def apply_text_timing_patch() -> None:
    from tribev2.eventstransforms import ExtractWordsFromAudio

    ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
        _patched_get_transcript_from_audio
    )
