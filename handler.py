import os
import sys
import shutil

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

if os.path.isdir("/runpod-volume"):
    CACHE_DIR = "/runpod-volume/hf_cache"
    print(f"[init] Volume found: {CACHE_DIR}")
    marker = os.path.join(CACHE_DIR, ".fastwan_ready_v3")
    if not os.path.exists(marker):
        print("[init] Wiping old data...")
        for item in os.listdir("/runpod-volume"):
            path = os.path.join("/runpod-volume", item)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as e:
                print(f"[init] skip {path}: {e}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        print("[init] Volume cleaned.")
else:
    CACHE_DIR = "/workspace/hf_cache"
    print(f"[init] No volume: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR

sys.stdout.flush()

import runpod

generator = None


def load_model():
    global generator
    if generator is not None:
        return generator

    print("[model] Loading FastWan 2.1 1.3B...")
    sys.stdout.flush()

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
    )

    marker_path = os.path.join(CACHE_DIR, ".fastwan_ready_v3")
    with open(marker_path, "w") as f:
        f.write("done")

    print("[model] Pipeline ready.")
    sys.stdout.flush()
    return generator


def handler(job):
    import uuid
    import base64

    try:
        job_input = job["input"]
        gen = load_model()

        prompt = job_input.get("prompt", "A beautiful sunset over the ocean")
        width = job_input.get("width", 480)
        height = job_input.get("height", 832)
        num_frames = job_input.get("num_frames", 61)
        num_inference_steps = job_input.get("num_inference_steps", 3)
        seed = job_input.get("seed", None)

        print(f"[infer] {width}x{height}, {num_frames}f, {num_inference_steps}steps")
        sys.stdout.flush()

        output_dir = f"/workspace/output_{uuid.uuid4()}"
        os.makedirs(output_dir, exist_ok=True)

        gen.generate_video(
            prompt,
            output_path=output_dir,
            save_video=True,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
        )

        video_path = None
        for f_name in os.listdir(output_dir):
            if f_name.endswith(".mp4"):
                video_path = os.path.join(output_dir, f_name)
                break

        if not video_path:
            return {"error": "No video generated"}

        print(f"[infer] Done -> {video_path}")
        sys.stdout.flush()

        bucket_url = os.environ.get("BUCKET_ENDPOINT_URL")
        if bucket_url:
            try:
                from runpod.serverless.utils import rp_upload
                video_url = rp_upload.upload_image(job["id"], video_path)
            except Exception:
                with open(video_path, "rb") as f:
                    video_url = f"data:video/mp4;base64,{base64.b64encode(f.read()).decode()}"
        else:
            with open(video_path, "rb") as f:
                video_url = f"data:video/mp4;base64,{base64.b64encode(f.read()).decode()}"

        import shutil as sh
        sh.rmtree(output_dir, ignore_errors=True)

        return {"video_url": video_url}

    except Exception as e:
        print(f"[error] {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return {"error": str(e)}


print("[init] Starting RunPod worker...")
sys.stdout.flush()
runpod.serverless.start({"handler": handler})
