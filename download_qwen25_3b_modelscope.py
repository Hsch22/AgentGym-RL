from modelscope import snapshot_download


TARGET_DIR = "/share/project/husicheng/muhan/AgentGym-RL/AgentGym-RL/models/Qwen2.5-3B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def main():
    model_dir = snapshot_download(MODEL_ID, local_dir=TARGET_DIR)
    print("DOWNLOAD_DONE", model_dir, flush=True)


if __name__ == "__main__":
    main()
