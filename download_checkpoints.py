import os
from huggingface_hub import snapshot_download


def main():
    download_dir = "checkpoints"
    os.makedirs(download_dir, exist_ok=True)

    snapshot_download(repo_id="SparkAudio/Spark-TTS-0.5B", cache_dir=download_dir)


if __name__ == "__main__":
    main()