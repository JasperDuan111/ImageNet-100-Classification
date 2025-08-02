import subprocess

def start_tensorboard(logdir="logs", port=6006):
    try:
        cmd = ["tensorboard", "--logdir", logdir, "--port", str(port)]
        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")

if __name__ == "__main__":
    start_tensorboard()
