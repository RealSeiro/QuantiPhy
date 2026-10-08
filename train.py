# QuantiPhy inference on Kaggle GPUs (2x T4) with an open-weight VLM.
# Push with `kaggle kernels push -p .`; predictions land in /kaggle/working/preds_<SPLIT>.csv
import os
import re
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# ───────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────
SPLIT = os.environ.get("QP_SPLIT", "validation")  # "validation" or "test"
MODEL_ID = os.environ.get("QP_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
NUM_FRAMES = int(os.environ.get("QP_FRAMES", "16"))
FRAME_WIDTH = int(os.environ.get("QP_WIDTH", "448"))
LIMIT = int(os.environ.get("QP_LIMIT", "0"))  # 0 = all rows
MAX_NEW_TOKENS = 48
OUT_DIR = "/kaggle/working"

DATASETS = {
    "validation": ("PaulineLi/QuantiPhy-validation", "validation_dataset.parquet", "validation_videos"),
    "test": ("PaulineLi/QuantiPhy", "test_dataset.parquet", ""),
}

print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print(f"SPLIT={SPLIT} MODEL={MODEL_ID} FRAMES={NUM_FRAMES} WIDTH={FRAME_WIDTH} LIMIT={LIMIT}")

# ───────────────────────────────────────────────────────────────
# Number parsing (copied from model_run_example/run_API_results.py)
# ───────────────────────────────────────────────────────────────
num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
EXACT_NUMBER_WITH_UNIT_RE = re.compile(
    r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*(?:"
    r"meters?|meter|m|kilometers?|kilometer|km|centimeters?|centimeter|cm|millimeters?|millimeter|mm|"
    r"inches?|inch|in|feet|foot|ft|yards?|yard|yd|"
    r"seconds?|second|s|minutes?|minute|min|hours?|hour|h|"
    r"m/s|m/s²|m/s^2|m/s2|km/h|kph|mph|ft/s|cm/s|mm/s|"
    r"m/s/s|m/s²|m/s^2|m/s2|ft/s²|ft/s^2|ft/s2|cm/s²|cm/s^2|cm/s2|mm/s²|mm/s^2|mm/s2|g"
    r")?\s*$",
    re.IGNORECASE,
)
UNITS_TO_REMOVE = [
    'kilometers', 'kilometer', 'km', 'meters', 'meter', 'm', 'centimeters', 'centimeter', 'cm',
    'millimeters', 'millimeter', 'mm', 'inches', 'inch', 'in', 'feet', 'foot', 'ft', 'yards', 'yard', 'yd',
    'hours', 'hour', 'h', 'minutes', 'minute', 'min', 'seconds', 'second', 's',
    'km/h', 'kph', 'mph', 'ft/s', 'm/s', 'cm/s', 'mm/s',
    'm/s/s', 'm/s²', 'm/s^2', 'm/s2', 'ft/s²', 'ft/s^2', 'ft/s2', 'cm/s²', 'cm/s^2', 'cm/s2',
    'mm/s²', 'mm/s^2', 'mm/s2', 'g', 'm^2', 'm²',
]


def parse_number(text):
    if not text or pd.isna(text):
        return np.nan
    original_text = str(text).strip()
    if EXACT_NUMBER_WITH_UNIT_RE.match(original_text):
        m = num_re.search(original_text)
        if m:
            return abs(float(m.group(0)))
    text_to_parse = original_text
    last_pos, last_delim = -1, None
    for d in ["Final Answer:", "Answer:", "=>", "=", ":", "is:", "The answer is:", "Example:"]:
        positions = [m.start() for m in re.finditer(re.escape(d), text_to_parse, re.IGNORECASE)]
        if positions and max(positions) > last_pos:
            last_pos, last_delim = max(positions), d
    if last_delim:
        parts = text_to_parse.rsplit(last_delim, 1)
        if len(parts) > 1:
            text_to_parse = parts[-1].strip()
    cleaned = text_to_parse.lower()
    for unit in UNITS_TO_REMOVE:
        cleaned = cleaned.replace(unit, '')
    for source in (cleaned, original_text):
        nums = num_re.findall(source)
        if nums:
            try:
                return abs(float(nums[-1]))
            except ValueError:
                pass
    return np.nan


# ───────────────────────────────────────────────────────────────
# Data
# ───────────────────────────────────────────────────────────────
repo_id, parquet_name, video_subdir = DATASETS[SPLIT]
data_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=f"/kaggle/temp/{SPLIT}")
df = pd.read_parquet(os.path.join(data_dir, parquet_name))
print("columns:", list(df.columns), "| rows:", len(df))
# The parquet uses short names; the prompt code and evaluator use the CSV names.
df = df.rename(columns={"prior": "ground_truth_prior", "answer": "ground_truth_posterior"})
if "ground_truth_prior" not in df.columns:
    raise KeyError(f"no prior column in {list(df.columns)}")

# The evaluator matches rows by the first column (an integer row ID).
id_col = next((c for c in df.columns if c in ("Unnamed: 0", "__index_level_0__", "id", "ID")), None)
if id_col is None:
    df = df.reset_index().rename(columns={"index": "Unnamed: 0"})
else:
    df = df.rename(columns={id_col: "Unnamed: 0"})
if LIMIT:
    df = df.head(LIMIT)

# Some filenames carry stray whitespace, so index videos by stripped stem.
video_root = os.path.join(data_dir, video_subdir)
video_index = {}
for dirpath, _, files in os.walk(video_root):
    for f in files:
        if f.lower().endswith(".mp4"):
            video_index[os.path.splitext(f.strip())[0].lower()] = os.path.join(dirpath, f)
print("videos found:", len(video_index))

_frame_cache = {}


def load_frames(video_id, fps_hint):
    """Uniformly sample NUM_FRAMES frames; returns (PIL frames, timestamps, duration, fps)."""
    if video_id in _frame_cache:
        return _frame_cache[video_id]
    path = video_index.get(str(video_id).strip().lower())
    if path is None:
        raise FileNotFoundError(f"video not found: {video_id}")
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        fps = float(fps_hint)
        if not fps > 0:
            raise ValueError
    except (TypeError, ValueError):
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idxs = np.linspace(0, max(total - 1, 0), num=min(NUM_FRAMES, max(total, 1))).round().astype(int)
    frames, stamps = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = FRAME_WIDTH / w
        frame = cv2.resize(frame, (FRAME_WIDTH, max(28, int(h * scale))), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        stamps.append(i / fps)
    cap.release()
    result = (frames, stamps, total / fps, fps)
    _frame_cache[video_id] = result
    return result


# ───────────────────────────────────────────────────────────────
# Model
# ───────────────────────────────────────────────────────────────
model_dir = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
processor = AutoProcessor.from_pretrained(model_dir)
model = AutoModelForImageTextToText.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
).eval()

SYSTEM_PROMPT = (
    "You are an expert video analyst specializing in physics measurements.\n"
    "Analyze the video frames carefully and provide ONLY the numerical answer with units. No explanation or reasoning needed.\n"
    "Format your response as: [value] [unit]\n"
    "Example: 2.5 cm\n"
    "Be as accurate as possible with measurements and calculations. Please give me an estimated answer even if you are not sure."
)


def build_messages(row, frames, stamps, duration, fps):
    content = [{"type": "text", "text": f"The video is {duration:.2f} s long, recorded at {fps:g} fps. "
                                        f"{len(frames)} frames are sampled uniformly with their timestamps.\n"}]
    for k, (img, t) in enumerate(zip(frames, stamps)):
        content.append({"type": "text", "text": f"Frame {k + 1} (t = {t:.3f} s):"})
        content.append({"type": "image", "image": img})
    prefix = []
    prior = row.get("ground_truth_prior")
    if prior is not None and not pd.isna(prior) and str(prior).strip():
        prefix.append(f"Given that {str(prior).strip()}.")
    video_type = str(row.get("video_type", ""))
    depth = row.get("depth_info")
    if len(video_type) >= 2 and video_type[1] == "3" and depth is not None and not pd.isna(depth) and str(depth).strip():
        prefix.append("Additionally, you have the following information about the distance between the objects "
                      f"in the video and the shooting camera: {str(depth).strip()}")
    context = " ".join(prefix) + "\n" if prefix else ""
    content.append({"type": "text", "text": f"\n{context}{row['question']}\n\n"
                                            "Please answer the question with numbers and units ONLY. No explanation needed."})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}]


@torch.inference_mode()
def generate(messages):
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


# ───────────────────────────────────────────────────────────────
# Run
# ───────────────────────────────────────────────────────────────
keep_cols = [c for c in ["Unnamed: 0", "video_id", "video_source", "video_type", "fps", "inference_type", "question",
                         "ground_truth_prior", "depth_info", "ground_truth_posterior"] if c in df.columns]
out_path = os.path.join(OUT_DIR, f"preds_{SPLIT}.csv")
rows, start = [], time.time()
for n, (_, row) in enumerate(df.iterrows(), start=1):
    record = {c: row[c] for c in keep_cols}
    try:
        frames, stamps, duration, fps = load_frames(row["video_id"], row.get("fps"))
        answer = generate(build_messages(row, frames, stamps, duration, fps))
    except Exception as e:  # keep going; the row counts as invalid
        answer = f"Error: {type(e).__name__}: {e}"
    record["raw_response"] = answer
    record["parsed_value"] = parse_number(answer) if not answer.startswith("Error:") else np.nan
    rows.append(record)
    gt = record.get("ground_truth_posterior", "")
    print(f"[{n}/{len(df)}] {row['video_id']} | {answer!r} -> {record['parsed_value']} | gt={gt} "
          f"| {time.time() - start:.0f}s", flush=True)
    if n % 20 == 0 or n == len(df):
        pd.DataFrame(rows).to_csv(out_path, index=False)

print("saved:", out_path)
