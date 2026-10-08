# SAM 3.1 smoke test on Kaggle T4: does it load, which precision works, how fast, are masks sane?
# Each precision mode runs in its own subprocess so autocast patches don't leak between runs.
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"
TEMP = "/kaggle/temp"
SAM_DIR = f"{TEMP}/sam3"
TESTS = [  # (video stem, SAM text prompt)
    ("internet_0004", "wood block"),
    ("simulation_0089", "bird"),
]


def setup():
    token = os.environ.get("HF_TOKEN")
    if not token:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    from huggingface_hub import hf_hub_download, login, snapshot_download
    login(token=token)
    if not os.path.isdir(SAM_DIR):
        subprocess.check_call(["git", "clone", "--depth", "1", "-q", "https://github.com/facebookresearch/sam3.git", SAM_DIR])
    # --no-deps: sam3 pins numpy<2, which would break Kaggle's preinstalled stack.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", SAM_DIR])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.17", "ftfy==6.1.1", "iopath>=0.1.10", "regex"])
    ckpt = hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt")
    data = snapshot_download("PaulineLi/QuantiPhy-validation", repo_type="dataset", local_dir=f"{TEMP}/validation")
    return ckpt, data


def run_mode(mode, ckpt, data):
    import numpy as np
    import torch

    # The SAM 3 code hard-codes bf16 autocast; T4 (sm75) has no native bf16, so remap it.
    if mode != "bf16":
        _orig = torch.autocast

        class _Autocast(_orig):
            def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
                if dtype == torch.bfloat16:
                    if mode == "fp32":
                        enabled = False
                    else:
                        dtype = torch.float16
                super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

        torch.autocast = _Autocast
        torch.amp.autocast = _Autocast

    import cv2
    from sam3.model_builder import build_sam3_predictor

    t0 = time.time()
    predictor = build_sam3_predictor(checkpoint_path=ckpt, version="sam3.1", use_fa3=False,
                                     compile=False, async_loading_frames=False)
    load_s = time.time() - t0
    report = {"mode": mode, "load_s": round(load_s, 1), "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "videos": []}

    for stem, prompt in TESTS:
        path = next(os.path.join(dp, f) for dp, _, fs in os.walk(data) for f in fs if f.strip().startswith(stem))
        t0 = time.time()
        sid = predictor.handle_request({"type": "start_session", "resource_path": path})["session_id"]
        first = predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": prompt})
        frames = {}
        for r in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid, "propagation_direction": "forward"}):
            o = r["outputs"]
            masks = np.asarray(o["out_binary_masks"])
            frames[r["frame_index"]] = {
                "obj_ids": np.asarray(o["out_obj_ids"]).tolist(),
                "boxes_xywh": np.round(np.asarray(o["out_boxes_xywh"]), 4).tolist(),
                "mask_area_px": [int(m.sum()) for m in masks],
            }
        elapsed = time.time() - t0
        predictor.handle_request({"type": "close_session", "session_id": sid})

        # Overlay the first object's mask on three frames so a human can check what SAM grabbed.
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        for fi in sorted({0, n // 2, max(n - 1, 0)} & set(frames)):
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            cap.release()
            if not ok:
                continue
            for (bx, by, bw, bh) in frames[fi]["boxes_xywh"]:
                x0, y0 = int(bx * w), int(by * h)  # boxes are relative coordinates
                cv2.rectangle(img, (x0, y0), (x0 + int(bw * w), y0 + int(bh * h)), (0, 255, 0), 2)
            cv2.imwrite(f"{WORK}/overlay_{mode}_{stem}_f{fi}.jpg", img)

        report["videos"].append({
            "video": stem, "prompt": prompt, "frames": n, "size_wh": [w, h],
            "seconds": round(elapsed, 1), "frames_with_object": sum(1 for f in frames.values() if f["obj_ids"]),
            "first_prompt_objects": np.asarray(first["outputs"]["out_obj_ids"]).tolist(),
            "sample": {k: frames[k] for k in sorted(frames)[:: max(1, len(frames) // 5)]},
        })
    report["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print("REPORT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_mode(sys.argv[1], sys.argv[2], sys.argv[3])
        sys.exit(0)
    ckpt, data = setup()
    import torch
    print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0), "| bf16 supported:", torch.cuda.is_bf16_supported())
    for mode in ["bf16", "fp16", "fp32"]:
        print(f"\n===== mode {mode} =====", flush=True)
        res = subprocess.run([sys.executable, __file__, mode, ckpt, data], capture_output=True, text=True, timeout=1800)
        lines = (res.stdout + res.stderr).splitlines()
        report = [l for l in lines if l.startswith("REPORT ")]
        print(report[0] if report else f"FAILED (exit {res.returncode}):\n" + "\n".join(lines[-25:]), flush=True)
