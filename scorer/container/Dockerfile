# syntax=docker/dockerfile:1
# Scoring container (see server.py). Built from the repo root so it can copy
# pipeline/sinemart: scorer/wrangler.jsonc sets image_build_context to "..".
# Versions pinned to the ones that embedded the live applicants
# (2026-10-27), so a photo scored here gets the pipeline's exact numbers.
FROM python:3.13-slim

# Behind an HTTPS proxy that re-signs traffic, pass its CA only for the build:
#   docker build --secret id=ca,src=<ca.crt> ...
# It is mounted per step and never stored in the image; without it the
# steps use the normal trust store.
ARG CA=/run/secrets/ca

# libEGL/GLES for the MediaPipe selfie segmenter on a headless box; libGL
# because MediaPipe brings its own (non-headless) OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=ca,required=false \
    if [ -f $CA ]; then export PIP_CERT=$CA; fi \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && pip install --no-cache-dir transformers==5.21.0 opencv-python-headless==5.0.0.93 mediapipe==1.1.1 \
       numpy==2.5.3 pillow==12.3.0

WORKDIR /app
ENV HF_HOME=/app/models/hf MODELS_DIR=/app/models PIPELINE_DIR=/app/pipeline PYTHONUNBUFFERED=1
RUN --mount=type=secret,id=ca,required=false \
    if [ -f $CA ]; then export CURL_CA_BUNDLE=$CA; fi \
    && mkdir -p /app/models \
    && curl -fsSL -o /app/models/face_detection_yunet_2023mar.onnx \
       https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    && curl -fsSL -o /app/models/selfie_segmenter.tflite \
       https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite

COPY pipeline/sinemart /app/pipeline/sinemart
COPY scorer/container/server.py /app/server.py
# Weights are baked in: the running container has no internet.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f $CA ]; then export SSL_CERT_FILE=$CA REQUESTS_CA_BUNDLE=$CA; fi \
    && python server.py --prep
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 8080
CMD ["python", "server.py"]
