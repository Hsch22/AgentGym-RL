from modelscope import snapshot_download
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET_DIR = ROOT / "AgentGym-RL" / "models" / "Qwen2.5-3B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def main():
    model_dir = snapshot_download(MODEL_ID, local_dir=str(TARGET_DIR))
    print("DOWNLOAD_DONE", model_dir, flush=True)


if __name__ == "__main__":
    main()
