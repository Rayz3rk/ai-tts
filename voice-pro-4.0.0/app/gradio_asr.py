

import shutil

from src.config import UserConfig

from app.abus_downloader import *
from app.abus_ffmpeg import *
from app.abus_demucs import *
from app.abus_genuine import *
from app.abus_files import *

from app.abus_asr_faster_whisper import *
from app.abus_asr_whisper import *
from app.abus_asr_whisper_timestamped import *

from src.i18n.i18n import I18nAuto
i18n = I18nAuto()


class GradioASR:
    def __init__(self, user_config: UserConfig):
        self.user_config = user_config
        
        self.fm = FileManager()
        
        self.downloader = YoutubeDownloader()
        
        asr_engine = self.user_config.get("asr_engine", 'faster-whisper')
        self.whisper_inf = self.switch_case(asr_engine)   
        
        # self.mdxnet_models_dir = os.path.join(os.getcwd(), 'model', 'mdxnet-model')
        # with open(os.path.join(self.mdxnet_models_dir, 'model_data.json')) as infile:
        #     self.mdx_model_params = json.load(infile)
    
    def switch_case(self, case):
        switch_dict = {
            'faster-whisper': lambda: FasterWhisperInference(),
            'whisper': lambda: WhisperInference(),
            'whisper-timestamped': lambda: WhisperTimestampedInference()
            }
        return switch_dict.get(case, lambda: FasterWhisperInference())()    
    
        

    def open_workspace_folder(self):
        cmd_open_explorer(path_workspace_folder())
    
    def open_temp_folder(self):
        cmd_open_explorer(path_gradio_folder())
    
    # def open_model_folder(self):
    #     cmd_open_explorer(self.mdxnet_models_dir)
        
    def get_asr_engines(self):
        return ['faster-whisper', 'whisper', 'whisper-timestamped']        
    
    def update_whisper_models(self, asr_engine):
        whisper_inf = self.switch_case(asr_engine)       
        model_list = whisper_inf.available_models()
        if len(model_list) > 0:
            model_name = self.user_config.get(f'{asr_engine.replace("-", "_")}_model', 'large')
            return gr.update(choices=model_list, value=model_name)
        
        return gr.update(choices=[], value=None)    
    
    
    def get_whisper_models(self):
        return self.whisper_inf.available_models()

        
    def get_whisper_languages(self):
        return self.whisper_inf.available_langs()
        
    def get_whisper_compute_types(self):
        return FasterWhisperInference.available_compute_types()
        
    # return Video, Audio, File    
    def upload_source(self, 
                      file_obj, mic_file, youtube_url: str, video_quality: str, audio_format: str):
        self.user_config.set("video_quality", video_quality)
        self.user_config.set("audio_format", audio_format)

        try:
            logger.debug(f'upload_source: file_obj={file_obj}, mic_file={mic_file}, youtube_url={youtube_url}')
            # ffmpeg는 다운로드 병합/오디오 추출에 필수 — 없으면 즉시 명확한 안내
            if shutil.which("ffmpeg") is None:
                raise gr.Error(i18n("ffmpeg is not installed. Run configure.bat (or configure.sh) as administrator to install it."), duration=None)

            self.fm = FileManager()
            if self._upload(file_obj, mic_file, youtube_url, video_quality, audio_format) == False:
                raise gr.Error(i18n("Please provide a media file, a microphone recording, or a YouTube URL."), duration=None)

            return self.fm.get_split("Source.video"), self.fm.get_split("Source.audio")
        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"[gradio_asr.py] upload_source - An error occurred: {e}")
            # duration=None: 사용자가 닫기 전까지 유지되는 오류 토스트 (자동 소멸로 놓치는 문제 방지)
            raise gr.Error(f'{e}', duration=None)

    def _upload(self,
                file_obj, mic_file, youtube_url: str, video_quality: str, audio_format: str):
        if (file_obj is not None):
            uploaded_file = cmd_copy_file_to(file_obj.name, path_workspace_subfolder(file_obj.name))
        elif mic_file and mic_file.strip():
            uploaded_file = cmd_copy_file_to(mic_file, path_workspace_subfolder(mic_file))
        elif youtube_url and youtube_url.strip():
            youtube_file = self.downloader.yt_download(youtube_url, path_youtube_folder(), video_quality)
            uploaded_file = cmd_copy_file_to(youtube_file, path_workspace_subfolder(youtube_file))
        else:
            return False
        
        self.source_file = uploaded_file
            
        
        self.has_audio, self.has_video = ffmpeg_codec_type(self.source_file)
        logger.debug(f'upload_source: source_file={self.source_file}, has_audio={self.has_audio}, has_video={self.has_video}')
        if self.has_audio == False:     # error
            raise gr.Error(i18n("The selected media has no audio track."), duration=None)
        elif self.has_video == False:   # audio-only
            self.fm.set_split("Source.video", None)
            self.fm.set_split("Source.audio", self.source_file)   
        else:
            input_audio_file = path_change_ext(self.source_file, f'.{audio_format}')
            ffmpeg_extract_audio(self.source_file, input_audio_file, audio_format)    
            self.fm.set_split("Source.video", self.source_file)
            self.fm.set_split("Source.audio", input_audio_file)
        return True
    
    def gradio_whisper_default(self):
        return ["large", "english", "float16", False, 0]
   
    def transcribe(self,
                  asr_engine, modelName, whisper_language, compute_type, highlight_words, denoise_level):
        self.user_config.set("asr_engine", asr_engine)
        self.user_config.set(f'{asr_engine.replace("-", "_")}_model', modelName)
        self.user_config.set("whisper_language", whisper_language)
        self.user_config.set("whisper_compute_type", compute_type)
        self.user_config.set("whisper_highlight_words", highlight_words)
        self.user_config.set("denoise_level", denoise_level)        
        
        try:
            source_audio = self.fm.get_split("Source.audio")
            # 미디어 미등록(또는 등록 진행 중) 상태에서 자막생성을 누른 경우
            if not source_audio:
                raise gr.Error(i18n("No media source registered. Please upload media and click Submit first."), duration=None)
            denoise_inst_path, denoise_vocal_path = self._denoise(source_audio, denoise_level)
            input_path = denoise_vocal_path if os.path.exists(denoise_vocal_path) else source_audio
            logger.debug(f'transcribe : input_path = {input_path}')
            
            params = WhisperParameters(model_size=modelName, 
                                       lang=whisper_language.lower(), 
                                       compute_type=compute_type)
            
            self.whisper_inf = self.switch_case(asr_engine)                            
            subtitles = self.whisper_inf.transcribe_file(input_path, params, highlight_words, gr.Progress())
            self.fm.set_subtitles(subtitles, whisper_language, source_audio) 
            srt_file = self.fm.get_subtitle('.srt')
            srt_string = self._read_subtitle_file(srt_file)
            logger.debug(f'srt_file = {srt_file}, self.source_file = {self.source_file}')
                        
            if(self.has_video and ffmpeg_browser_compatible(self.source_file)):
                if srt_file:
                    # gradio 6: 자막은 (video, srt) 튜플 대신 Video 컴포넌트의 subtitles 속성으로 전달
                    return gr.Video(value=self.source_file, subtitles=srt_file), srt_string, self.fm.get_all_files()
                else:
                    return self.source_file, srt_string, self.fm.get_all_files()
            else:
                return None, srt_string, self.fm.get_all_files()
            
        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"[gradio_asr.py] transcribe - An error occurred: {e}")
            raise gr.Error(f'{e}', duration=None)
            return None, None, None            
                
    # return inst, vocal    
    def _denoise(self, source_audio, denoise_level=2):
        if denoise_level == 1:
            return self._demucs_htdemucs(source_audio)
        elif denoise_level ==2:
            return self._demucs_htdemucs_ft(source_audio)
        else:
            return "", ""
                
        
            
    def _demucs_htdemucs(self, source_audio):
        _, extension = os.path.splitext(os.path.basename(source_audio))
        output_dir = os.path.dirname(source_audio)
        
        inst_audio_file, vocal_audio_file = demucs_split_file(source_audio, output_dir, 'htdemucs', extension[1:])
        self.fm.set_split("Instrumental.audio", inst_audio_file)
        self.fm.set_split("Vocals.audio", vocal_audio_file)

        return inst_audio_file, vocal_audio_file
    
    def _demucs_htdemucs_ft(self, source_audio):
        _, extension = os.path.splitext(os.path.basename(source_audio))
        output_dir = os.path.dirname(source_audio)
        
        inst_audio_file, vocal_audio_file = demucs_split_file(source_audio, output_dir, 'htdemucs_ft', extension[1:])
        self.fm.set_split("Instrumental.audio", inst_audio_file)
        self.fm.set_split("Vocals.audio", vocal_audio_file)

        return inst_audio_file, vocal_audio_file    
    
    def _read_subtitle_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content    

        