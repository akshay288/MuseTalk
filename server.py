import os
import uuid
import subprocess
import yaml
import requests
import runpod
from timeit import default_timer


UPLOAD_FOLDER = '/tmp/vidai-files'
CONFIG_FOLDER = '/tmp/vidai-configs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONFIG_FOLDER, exist_ok=True)
os.makedirs("./results/avatars", exist_ok=True)


def upload_file_to_s3(presigned_url, file_path):
    with open(file_path, 'rb') as file:
        response = requests.put(presigned_url, data=file)
        if response.status_code == 200:
            print("File uploaded successfully.")
        else:
            print(f"Failed to upload file. Status code: {response.status_code}, Response: {response.text}")


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"File downloaded successfully and saved to {save_path}")
    else:
        raise Exception(f"Failed to download file. HTTP status code: {response.status_code}")


def run_script(job_id, upload_url, video_path, audio_path, avatar_id, bbox_shift):
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

    output_path = f"results/avatars/{avatar_id}/vid_output/audio_{job_id}.mp4"
    if os.path.exists(output_path):
        upload_file_to_s3(upload_url, output_path)
    else:
        print("Could not find output path")


def handler(job):
    data = job["input"]

    upload_url = data.get("uploadUrl")
    video_path = data.get("video")
    audio_path = data.get("audio")
    avatar_id = data.get("avatar")
    avatar_pretrained = data.get("avatarPretrained")
    bbox_shift = data.get("bboxShift", 0)

    if not video_path:
        return { "error": "Must supply video path" }
    if not audio_path:
        return { "error": "Must supply audio path" }
    if not avatar_id:
        return { "error": "Must supply avatar id" }
    if not upload_url:
        return { "error": "Must supply upload url" }

    job_id = str(uuid.uuid4())
    print("--------------------------")
    print(job_id)
    print(upload_url)
    print(video_path)
    print(audio_path)
    print(avatar_id)
    print(avatar_pretrained)
    print(bbox_shift)
    print("--------------------------")

    if not os.path.exists(f"./results/avatars/{avatar_id}") and avatar_pretrained:
        print("downloading pretrained")
        os.makedirs(f"./results/tmp/download", exist_ok=True)
        tar_path = f"./results/tmp/download/{avatar_id}.tar"
        start_download = default_timer()
        download_file(avatar_pretrained, tar_path)
        end_download = default_timer()

        start_tar = default_timer()
        subprocess.run(["tar", "-xf", tar_path, "-C", f"./results/tmp/download"])
        subprocess.run(["cp", "-r", f"./results/tmp/download/results/avatars/{avatar_id}", f"./results/avatars/{avatar_id}"])
        subprocess.run(["rm", "-rf", "./results/tmp/download"])
        end_tar = default_timer()

        print(f"done downloading pretrained - Download Time ({end_download - start_download}) Tar Time ({end_tar - start_tar})")

    run_script(job_id, upload_url, video_path, audio_path, avatar_id, bbox_shift)

    return { "success": True }


runpod.serverless.start({"handler": handler})
