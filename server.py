import os
import uuid
import subprocess
import yaml
import requests
import runpod


UPLOAD_FOLDER = '/tmp/vidai-files'
CONFIG_FOLDER = '/tmp/vidai-configs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONFIG_FOLDER, exist_ok=True)


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Video downloaded successfully and saved to {save_path}")
    else:
        raise Exception(f"Failed to download video. HTTP status code: {response.status_code}")


def run_script(job_id, video_path, audio_path, avatar_id, bbox_shift):
    os.makedirs(f"{UPLOAD_FOLDER}/{job_id}", exist_ok=True)
    os.makedirs(f"{UPLOAD_FOLDER}/avatars/{avatar_id}", exist_ok=True)
    avatar_local_path = f"{UPLOAD_FOLDER}/avatars/{avatar_id}/video.mp4"
    audio_local_path = f"{UPLOAD_FOLDER}/{job_id}/audio.mp3"

    download_file(audio_path, audio_local_path)
    if not os.path.exists(avatar_local_path):
        download_file(video_path, avatar_local_path)

    config_path = f"{CONFIG_FOLDER}/{job_id}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(
            {
                avatar_id: {
                    "preparation": True,
                    "bbox_shift": int(bbox_shift) if bbox_shift else 0,
                    "video_path": avatar_local_path,
                    "audio_clips": {
                        f"audio_{job_id}": audio_local_path,
                    },
                }
            },
            f,
            default_flow_style=False
        )

    subprocess.run([
        "python3", "-m", "scripts.realtime_inference",
        "--inference_config", config_path, "--batch_size", "4"
    ])


def handler(job):
    data = job["input"]

    video_path = data.get("video")
    audio_path = data.get("audio")
    avatar_id = data.get("avatar")
    bbox_shift = data.get("bboxShift", 0)

    if not video_path:
        return { "error": "Must supply video path" }
    if not audio_path:
        return { "error": "Must supply audio path" }
    if not avatar_id:
        return { "error": "Must supply avatar id" }

    job_id = str(uuid.uuid4())
    run_script(job_id, video_path, audio_path, avatar_id, bbox_shift)

    return { "success": True }


runpod.serverless.start({"handler": handler})
