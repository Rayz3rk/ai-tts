import gradio as gr
import pysubs2
import re
import time
from deep_translator import GoogleTranslator

from app.abus_genuine import *
from app.abus_path import *
from app.abus_text import *
from app.abus_nlp_spacy import *

import structlog
logger = structlog.get_logger()

# 회사망 등의 보안장비가 translate.google.com 요청을 간헐적으로 거부(레이트리밋)하는
# 환경 대응: 요청 간 소량 간격 + 거부 시 지수 백오프 재시도.
REQUEST_INTERVAL = 0.2   # 각 요청 사이 간격(초) — 레이트리밋 유발 완화
MAX_RETRIES = 4          # 라인당 재시도 횟수 (백오프: 2, 4, 8초)


class DeepTranslator:
    def __init__(self) -> None:
        self.translator = GoogleTranslator(source='auto', target='en')
        self.languages_dict = GoogleTranslator().get_supported_languages(as_dict=True)
        
   
    def get_languages(self) -> list:
        capitalized_keys = [key.capitalize() for key in self.languages_dict.keys()]
        return capitalized_keys
    
    def get_language_code(self, language_name) -> str:
        search_name = language_name.lower()
        for key, value in self.languages_dict.items():
            if key.lower() == search_name:
                return value
        return "en"
    
    def get_language_value(self, language_name):
        search_name = language_name.lower()
        for key, value in self.languages_dict.items():
            if key.lower() == search_name:
                return key
        return None


    @staticmethod
    def _translate_with_retry(translator: GoogleTranslator, text: str) -> str:
        """레이트리밋으로 인한 간헐적 연결 거부를 지수 백오프로 흡수한다."""
        delay = 2
        for attempt in range(MAX_RETRIES):
            try:
                result = translator.translate(text)
                time.sleep(REQUEST_INTERVAL)
                return result
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logger.warning(f"[abus_translate_deep.py] translate retry {attempt + 1}/{MAX_RETRIES - 1} in {delay}s - {e}")
                time.sleep(delay)
                delay *= 2
    
  
    
    def translate_text(self, source_lang: str, target_lang: str, text: str, progress=gr.Progress()) -> str:
        source_code = self.get_language_code(source_lang)
        target_code = self.get_language_code(target_lang)
        
        self.translator.source = source_code
        self.translator.target = target_code
        
        # line 끝 마침표 확인인
        use_punctuation = AbusText.has_ending_marks([text])
        
        # 텍스트를 문장 단위로 분리
        sentences = AbusText.split_into_sentences(text, use_punctuation)
        sentences = sentences
        
        translated_sentences = []
        failed_count = 0

        # 각 문장을 번역
        for sentence in progress.tqdm(sentences, desc="Translating sentences..."):
            try:
                translated = self._translate_with_retry(self.translator, sentence)
                translated_sentences.append(translated)
                logger.debug(f"[abus_translate_deep.py] translate_text - {source_code}: {sentence} -> {target_code}: {translated}")
            except Exception as e:
                logger.error(f"Translation error: {e}")
                translated_sentences.append(sentence)  # 에러 발생 시 원본 문장 사용
                failed_count += 1

        if failed_count > 0:
            gr.Warning(f"({failed_count}/{len(sentences)}) " + i18n("Some lines could not be translated and were kept in the original language. The translation service may be rate-limited by your network - please try again later."), duration=None)

        # 번역된 문장들을 다시 하나의 텍스트로 결합
        final_text = ' '.join(translated_sentences)
        return final_text

    def translate_file(self, source_lang: str, target_lang: str, subtitle_file_path: str, output_file_path: str, progress=gr.Progress()):
        tts_source_file = path_add_postfix(subtitle_file_path, f"-{source_lang}", ".srt")
        
        # AbusText.process_subtitle_for_tts(subtitle_file_path, tts_source_file)
        AbusSpacy.process_subtitle_for_tts(subtitle_file_path, tts_source_file)
        
        source_code = self.get_language_code(source_lang)
        target_code = self.get_language_code(target_lang)

        
        translator = GoogleTranslator(source=source_code, target=target_code)
        logger.debug(f"[abus_translate_deep.py] translate_file {source_code}: {subtitle_file_path} -> {target_code}: {output_file_path}")

        # Load subtitles using pysubs2
        full_subs = pysubs2.load(tts_source_file)
        subs = full_subs
        
        # 구두점이 없는 언어의 경우 각 자막을 개별적으로 번역
        failed_count = 0
        for event in progress.tqdm(subs, desc='Translate...'):
            if not event.text:
                continue

            text = event.plaintext
            try:
                translated_text = self._translate_with_retry(translator, text)
                if translated_text:
                    event.text = translated_text
                    logger.debug(f"[abus_translate_deep.py] translate_file : text       - {text}")
                    logger.debug(f"[abus_translate_deep.py] translate_file : translated - {translated_text}")
                else:
                    logger.warning(f"[abus_translate_deep.py] translate_file - Empty translation for: {text}")
            except Exception as e:
                logger.error(f"Translation error for text '{text}': {e}")
                # 에러 발생 시 원본 텍스트 유지
                failed_count += 1

        if failed_count > 0:
            gr.Warning(f"({failed_count}/{len(subs)}) " + i18n("Some lines could not be translated and were kept in the original language. The translation service may be rate-limited by your network - please try again later."), duration=None)

        # Save the translated subtitles
        subs.save(output_file_path)
        cmd_delete_file(tts_source_file)

            
