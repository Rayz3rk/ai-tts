FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg espeak-ng && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY audio_studio ./audio_studio
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple '.[tts,singing]' https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
ENV AUDIO_STUDIO_DATA=/data
EXPOSE 7861
CMD ["audio-studio", "serve", "--host", "0.0.0.0", "--port", "7861", "--device", "cpu"]
