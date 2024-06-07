import os
import uuid
import yaml
import subprocess
import sys
import shutil
from tqdm import tqdm
import boto3
from dotenv import load_dotenv
load_dotenv()

S3_CLIENT = boto3.client('s3')
BUCKET_NAME = "vidai-assets"

UPLOAD_FOLDER = '/tmp/vidai-files'
CONFIG_FOLDER = '/tmp/vidai-configs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONFIG_FOLDER, exist_ok=True)

avatars = [
	{
		"ID": "f093a357-aab4-44e4-b50a-8203a7b25f75",
		"Gender": "female",
		"BBOXShift": 0,
	},
	{
		"ID": "f33eb471-e8b6-4788-ba50-8d8815d05d6d",
		"Gender": "male",
		"BBOXShift": 0,
	},
	{
		"ID": "ec3ce70a-5554-4198-92f8-50dc0a77635f",
		"Gender": "male",
		"BBOXShift": -7,
	},
	{
		"ID": "c443a8d3-6890-4c17-b504-33cf89033696",
		"Gender": "female",
		"BBOXShift": 0,
	},
	{
		"ID": "bbcd22b3-2ee4-4eb1-a1c9-33f5d819f2e5",
		"Gender": "female",
		"BBOXShift": 0,
	},
	{
		"ID": "aa2d389c-7fd0-4f96-a6a8-e8083d2b5978",
		"Gender": "female",
		"BBOXShift": 0,
	},
	{
		"ID": "8e0851cd-908e-4f69-8798-58db2804ad2c",
		"Gender": "female",
		"BBOXShift": 0,
	},
	{
		"ID": "15447160-224d-4522-be4a-9b7f0a7eda41",
		"Gender": "male",
		"BBOXShift": 9,
	},
	{
		"ID": "cdb2f06a-b893-425d-b48b-17d5babbd1fe",
		"Gender": "male",
		"BBOXShift": 29,
	},
	{
		"ID": "eb479db9-2fdf-4b86-8680-7494f95308f5",
		"Gender": "male",
		"BBOXShift": 25,
	},
	{
		"ID": "dfd478ae-ec4c-4114-b2f2-2eb6a61ba2fd",
		"Gender": "male",
		"BBOXShift": 10,
	},
	{
		"ID": "0aa9ee8b-82e0-4dc0-b92c-8de04b785796",
		"Gender": "male",
		"BBOXShift": 10,
	},
	{
		"ID": "91b798ee-f1a5-4f73-8804-8bb4a3a83938",
		"Gender": "male",
		"BBOXShift": 33,
	},
	{
		"ID": "1ba18feb-c6e5-4fbb-93d4-9420a67155c4",
		"Gender": "male",
		"BBOXShift": 25,
	},
	{
		"ID": "92be7247-669f-4322-b4f3-4fb3942737bf",
		"Gender": "female",
		"BBOXShift": 0,
	},
	{
		"ID": "062f2380-4df0-45d9-8ede-834f020b6b35",
		"Gender": "male",
		"BBOXShift": 35,
	},
	{
		"ID": "7bb04719-6669-4d93-985a-715f032b6a64",
		"Gender": "female",
		"BBOXShift": 30,
	},
]

def train(avatar_id, gender, bbox_shift):
    job_id = str(uuid.uuid4())
    os.makedirs(f"{UPLOAD_FOLDER}/{job_id}", exist_ok=True)
    os.makedirs(f"{UPLOAD_FOLDER}/avatars/{avatar_id}", exist_ok=True)
    avatar_local_path = f"{UPLOAD_FOLDER}/avatars/{avatar_id}/video.mp4"
    audio_local_path = f"{UPLOAD_FOLDER}/{job_id}/audio.mp3"

    shutil.copyfile(f"data/audio/{gender}_audio.wav", audio_local_path)
    if not os.path.exists(avatar_local_path):
        S3_CLIENT.download_file(BUCKET_NAME, f"common/avatar/{avatar_id}/video_1280.mp4", avatar_local_path)

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
    if not os.path.exists(output_path):
        print(f"Could not find output path for avatar {avatar_id}")


def train_job():
    for avatar in tqdm(avatars):
        train(avatar["ID"], avatar["Gender"], avatar["BBOXShift"])
	

def bundle_job():
    for avatar in tqdm(avatars):
        avatarID = avatar["ID"]
        subprocess.run(
            ["tar", "-cvf", f"results/avatars/{avatarID}.tar", f"results/avatars/{avatarID}"],
            stdout=subprocess.DEVNULL,
        )


def upload_to_s3(file_path, object_name):
    try:
        S3_CLIENT.upload_file(file_path, BUCKET_NAME, object_name)
        print(f"Uploaded {object_name} to S3 bucket {BUCKET_NAME}")
    except Exception as e:
        print(f"Failed to upload {object_name}: {e}")


def push_job():
    for avatar in tqdm(avatars):
        avatarID = avatar["ID"]
        upload_to_s3(f"results/avatars/{avatarID}.tar", f"common/avatar/{avatarID}/pretrained/video_1280.tar")


if __name__ == "__main__":
    if sys.argv[1] == "bundle":
        bundle_job()
    elif sys.argv[1] == "push":
        push_job()
    else:
        train_job()