FROM nvidia/cuda:12.2.2-base-ubuntu22.04

WORKDIR /app

RUN apt update

RUN apt-get install -y python3 python3-pip curl ffmpeg libsm6 libxext6 sed -y
RUN apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY scripts scripts
COPY musetalk musetalk
COPY assets assets
COPY data data
COPY configs configs
COPY requirements.txt requirements.txt
COPY server.py server.py

RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.1"
RUN mim install "mmdet>=3.1.0"
RUN mim install "mmpose>=1.1.0"

RUN curl https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz > ffmpeg-git-amd64-static.tar.xz
RUN tar xf ffmpeg-git-amd64-static.tar.xz
ENV FFMPEG_PATH /app/ffmpeg-git-20240504-amd64-static

RUN curl -O https://vidai-docker-data-public.s3.us-west-2.amazonaws.com/models.tar && tar -xf models.tar && rm models.tar
RUN sed -i '20s/.*/    # removed!/' /usr/local/lib/python3.10/dist-packages/mmpose/datasets/builder.py

CMD ["python3", "-u", "server.py"]
