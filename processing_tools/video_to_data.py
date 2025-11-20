# save_frames_memmap.py
import cv2
import numpy as np
from pathlib import Path
import json

# ---------------------------
# PARAMETERS
# ---------------------------
video_folder = Path("data/raw_video")
output_folder = Path("data/frame_arrays")
output_folder.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (224, 224)  # width, height for CNN
VIDEO_LIST = list(video_folder.glob("*.mp4"))  # or .avi
dtype = np.uint8

# ---------------------------
# PROCESS EACH VIDEO
# ---------------------------
for video_path in VIDEO_LIST:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        continue

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h = TARGET_SIZE

    # Create memmap on disk
    memmap_path = output_folder / f"{video_path.stem}.dat"
    frames = np.memmap(memmap_path, dtype=dtype, mode="w+", shape=(num_frames, out_h, out_w, 3))

    print(f"Processing {video_path.name}: {num_frames} frames -> {TARGET_SIZE}")

    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame read failed at idx {idx}")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
        # Store in memmap
        frames[idx] = frame_resized

        if idx % 100 == 0:
            print(f"Processed {idx}/{num_frames} frames")

    frames.flush()  # ensure data written to disk
    cap.release()
    print(f"Finished {video_path.name}, saved to {memmap_path}")

    # ---------------------------
    # SAVE METADATA
    # ---------------------------
    meta = {
        "filename": str(memmap_path),
        "num_frames": num_frames,
        "height": out_h,
        "width": out_w,
        "channels": 3,
        "dtype": str(dtype)
    }
    meta_path = output_folder / f"{video_path.stem}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"Saved metadata to {meta_path}")
