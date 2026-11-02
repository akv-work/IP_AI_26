import gc
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ultralytics import YOLO

model = YOLO("yolo12m.pt")
source_video = "/content/drive/MyDrive/SHARED/One Step Closer.mp4"

def run_and_get_stats_advanced(tracker_name, config_file):
    id_lifespans = {}
    confidences = []
    frames_with_detections = 0
    total_frames = 0
    total_detections = 0
    cumulative_ids = []
    seen_so_far = set()

    start_time = time.time()

    results = model.track(
        source=source_video,
        tracker=config_file,
        classes=[0],
        save=True,
        name=tracker_name,
        exist_ok=True,
        stream=True,
    )

    for r in results:
        total_frames += 1

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            confs = r.boxes.conf.tolist()
            seen_so_far.update(ids)

            if len(ids) > 0:
                frames_with_detections += 1
                total_detections += len(ids)
                confidences.extend(confs)

                for obj_id in ids:
                    id_lifespans[obj_id] = id_lifespans.get(obj_id, 0) + 1

        cumulative_ids.append(len(seen_so_far))

    total_time = time.time() - start_time
    fps = total_frames / total_time if total_time > 0 else 0

    lifespans = list(id_lifespans.values()) if id_lifespans else [0]
    total_unique_ids = len(id_lifespans)

    short_tracks = sum(1 for l in lifespans if l < 10)
    short_tracks_pct = (
        (short_tracks / total_unique_ids * 100) if total_unique_ids > 0 else 0
    )

    stats = {
        "Tracker": tracker_name,
        "Frames": total_frames,
        "FPS": round(fps, 2),
        "Uniq ID": total_unique_ids,
        "Avg. detections per frame": round(
            total_detections / total_frames if total_frames else 0, 2
        ),
        "Avg. ID ttl": round(float(np.mean(lifespans)), 1),
        "Median ttl ID": round(float(np.median(lifespans)), 1),
        "Max ttl ID": max(lifespans),
        "Short tracks (<10 frames)": f"{short_tracks} ({short_tracks_pct:.1f}%)",
        "Avg. confidence": (
            round(float(np.mean(confidences)), 3) if confidences else 0
        ),
    }

    del results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats, id_lifespans, cumulative_ids

path_botsort = (
    "custom_botsort.yaml"
)
path_bytetrack = (
    "custom_bytetrack.yaml"
)

stats_botsort, lifespans_botsort, cumulative_botsort = (
    run_and_get_stats_advanced("botsort", path_botsort)
)

stats_bytetrack, lifespans_bytetrack, cumulative_bytetrack = (
    run_and_get_stats_advanced("bytetrack", path_bytetrack)
)

df = pd.DataFrame([stats_botsort, stats_bytetrack])
display(df.T)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    "i love linkin park",
    fontsize=16,
    fontweight="bold",
)

trackers = ["BoT-SORT", "ByteTrack"]

unique_ids_count = [stats_botsort["Uniq ID"], stats_bytetrack["Uniq ID"]]
bars = axes[0, 0].bar(
    trackers, unique_ids_count, color=["#4C72B0", "#55A868"], width=0.4
)
axes[0, 0].set_title("ID`s count")
axes[0, 0].set_ylabel("ID`s count")
for bar in bars:
    yval = bar.get_height()
    axes[0, 0].text(
        bar.get_x() + bar.get_width() / 2,
        yval + 5,
        int(yval),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

axes[0, 1].hist(
    list(lifespans_botsort.values()),
    bins=30,
    alpha=0.6,
    label="BoT-SORT",
    color="#4C72B0",
    range=(0, 200),
)
axes[0, 1].hist(
    list(lifespans_bytetrack.values()),
    bins=30,
    alpha=0.6,
    label="ByteTrack",
    color="#55A868",
    range=(0, 200),
)
axes[0, 1].set_title("ID`s ttl")
axes[0, 1].set_xlabel("Track (Long)")
axes[0, 1].set_ylabel("Track (Count)")
axes[0, 1].legend()

fps_values = [
    stats_botsort["FPS"],
    stats_bytetrack["FPS"],
]
bars_fps = axes[1, 0].bar(
    trackers, fps_values, color=["#C44E52", "#8172B0"], width=0.4
)
axes[1, 0].set_title("FPS")
axes[1, 0].set_ylabel("FPS")
for bar in bars_fps:
    yval = bar.get_height()
    axes[1, 0].text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1,
        f"{yval:.1f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

axes[1, 1].plot(cumulative_botsort, label="BoT-SORT", color="#4C72B0", linewidth=2)
axes[1, 1].plot(
    cumulative_bytetrack, label="ByteTrack", color="#55A868", linewidth=2
)
axes[1, 1].set_title("New ID Dynamics")
axes[1, 1].set_xlabel("Frame number")
axes[1, 1].set_ylabel("Summ. ID")
axes[1, 1].legend()

plt.tight_layout()
plt.show()
