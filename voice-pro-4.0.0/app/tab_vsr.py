import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import gradio as gr
from src.config import UserConfig

import src.ui as ui
from src.i18n.i18n import I18nAuto
i18n = I18nAuto()

from app.abus_genuine import *
from app.gradio_vsr import *


def vsr_tab(user_config: UserConfig):
    vsr = GradioVSR(user_config)
    
    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                gr.HTML(f'<center><h4>{i18n("Upload media")}</h4></center>')
                original_file = gr.File(label=i18n("Upload File"), type="filepath", file_count="single", file_types=['audio', 'video'])
                with gr.Group():
                    url_text = gr.Textbox(label=i18n("YouTube URL"), placeholder="https://youtu.be/abcdefgh...")
                    youtube_quality_radio = gr.Radio(label=i18n("YouTube Video Quality"), choices=["low", "good", "best"], value=user_config.get("video_quality", "good"))
                    youtube_format_radio = gr.Radio(label=i18n("Audio Format"), choices=["wav", "flac", "mp3"], value=user_config.get("audio_format", "flac"))                    
            with gr.Row():
                clear_button = gr.ClearButton(value=i18n("Clear"))
                submit_button = gr.Button(value=i18n("Submit"), variant="primary")
        with gr.Column(scale=8):
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        gr.HTML(f'<center><h4>{i18n("Input Video")}</h4></center>')
                        input_video = gr.Video(label=i18n("Video"), interactive=False)
                with gr.Column():
                    with gr.Group():
                        gr.HTML(f'<center><h4>{i18n("RTX Video")}</h4></center>')    
                        output_video = gr.Video(label=i18n("Video"), interactive=False)
            output_files = gr.File(label=i18n("Files"), type="filepath", file_count="multiple", interactive=False)
            with gr.Row():
                workspace_button = gr.Button(value=i18n("🗂️ Open workspace folder"), variant="secondary")
                temp_button = gr.Button(value=i18n("🗀 Open Temp folder"), variant="secondary")                        
                
        with gr.Column(scale=3):
            with gr.Group():
                gr.HTML(f'<center><h4>{i18n("NVIDIA RTX")}</h4></center>')            
            with gr.Group():
                gr.HTML(f'<center><h6>{i18n("Artifact Reduction")}</h6></center>')
                var_enable_check = gr.Checkbox(label=i18n("Enable"), info="", value=user_config.get("var_enable", False))
                var_mode_radio = gr.Radio(label=i18n("mode"), choices=[0, 1], value=user_config.get("var_mode", 0), 
                                          info=i18n("Mode 0 removes lesser artifacts, preserves low gradient information better, and is suited for higher bitrate videos."))
            with gr.Group():
                gr.HTML(f'<center><h6>{i18n("Super Resolution")}</h6></center>')
                vsr_enable_check = gr.Checkbox(label=i18n("Enable"), info="", value=user_config.get("vsr_enable", True))
                vsr_mode_radio = gr.Radio(label=i18n("mode"), choices=[0, 1], value=user_config.get("vsr_mode", 0),
                                          info=i18n("Mode 0 enhances less and removes more encoding artifacts and is suited for lower-quality videos."))
                vsr_scale_radio = gr.Radio(label=i18n("Scale Factor"), choices=[1.5, 2, 3, 4], value=user_config.get("vsr_scale", 2))             
            with gr.Group():
                gr.HTML(f'<center><h6>{i18n("Compression")}</h6></center>')
                compression_enable_check = gr.Checkbox(label=i18n("Enable"), info="", value=user_config.get("compression_enable", True))
                compression_crf_slider = gr.Slider(0, 51, step=1, label=i18n("CRF(Constant Rate Factor)"), info=i18n("Adjust compression quality (0-51, lower is better quality. Default is 23)"), 
                                                   value=user_config.get("compression_crf", 23))
                compression_preset_dropdown = gr.Dropdown(
                    ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], label=i18n("Preset"), info=i18n("Adjusting Compression Speed and Efficiency"),
                    value=user_config.get("compression_preset", "medium"))
            with gr.Row():
                default_button = gr.ClearButton(value=i18n("Load Defaults"))
                rtx_button = gr.Button(value=i18n("RTX"), variant="primary")          
                        
    clear_button.add([original_file, url_text])
    submit_button.click(vsr.upload_source, 
                            inputs=[original_file, url_text, youtube_quality_radio, youtube_format_radio], 
                            outputs=[input_video, output_files])
    
    workspace_button.click(vsr.open_workspace_folder)
    temp_button.click(vsr.open_temp_folder)
        
    default_button.click(vsr.gradio_default_rtx,
                    outputs=[var_enable_check, 
                             var_mode_radio, 
                             vsr_enable_check, 
                             vsr_mode_radio, 
                             vsr_scale_radio, 
                             compression_enable_check, 
                             compression_crf_slider, 
                             compression_preset_dropdown])
    rtx_button.click(vsr.run_rtx, 
                        inputs=[var_enable_check,
                                var_mode_radio,
                                vsr_enable_check,
                                vsr_mode_radio, 
                                vsr_scale_radio,
                                compression_enable_check,
                                compression_crf_slider,
                                compression_preset_dropdown], 
                        outputs=[output_video, 
                                 output_files])
    