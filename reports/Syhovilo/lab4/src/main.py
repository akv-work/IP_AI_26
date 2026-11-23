"""ЛР4 (Суховило В. С.): трекинг нескольких жестов YOLO26m + BoT-SORT/ByteTrack.

Режим analyze собирает показатели из реальных CSV без повторной обработки видео.
Режим run выполняет эксперименты при наличии best.pt и видеозаписей.

Примеры:
  python main.py --base /content/drive/MyDrive/OIIS_LR4_Suhovilo --mode analyze
  python main.py --base /content/drive/MyDrive/OIIS_LR4_Suhovilo --mode run --device 0
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

FIRST_EXPERIMENTS = [
    "botsort", "bytetrack", "botsort_tuned", "bytetrack_tuned",
    "botsort_buffer60", "botsort_high018", "botsort_new018",
    "bytetrack_buffer60", "bytetrack_high018", "bytetrack_new018",
]
SECOND_EXPERIMENTS = [
    "internet_botsort", "internet_bytetrack",
    "internet_botsort_tuned", "internet_bytetrack_tuned",
]
ALL_EXPERIMENTS = FIRST_EXPERIMENTS + SECOND_EXPERIMENTS
BASES = ["botsort", "bytetrack"]
MODES = {
    "tuned": {"track_high_thresh": 0.18, "new_track_thresh": 0.18, "track_buffer": 60},
    "buffer60": {"track_buffer": 60},
    "high018": {"track_high_thresh": 0.18},
    "new018": {"new_track_thresh": 0.18},
}


def read_data(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    required = {"frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"}
    if not required.issubset(df.columns):
        raise ValueError(f"Не хватает столбцов в {csv_file.name}: {sorted(required - set(df.columns))}")
    return df


def get_frame_counts(results: Path) -> dict[str, int]:
    # Дополнительные значения сохранены из фактически выполненных прогонов.
    metadata = results / "metadata.json"
    if metadata.is_file():
        data = json.loads(metadata.read_text(encoding="utf-8"))
        values = data.get("source_frames", {})
        if "own" in values and "internet" in values:
            return {"own": int(values["own"]), "internet": int(values["internet"])}

    reference = results / "final_comparison.csv"
    own = None
    if reference.is_file():
        old = pd.read_csv(reference)
        if "frames" in old.columns and len(old):
            own = int(old.iloc[0]["frames"])
    try:
        import cv2
    except ImportError:
        cv2 = None
    counts = {}
    for source, name in [("own", "botsort_result.mp4"), ("internet", "internet_botsort_result.mp4")]:
        if cv2 and (results / name).is_file():
            cap = cv2.VideoCapture(str(results / name))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frames > 0:
                counts[source] = frames
    if "own" not in counts and own is not None:
        counts["own"] = own
    if set(counts) != {"own", "internet"}:
        raise FileNotFoundError("Нужны metadata.json с source_frames, либо исходные MP4 обоих видео")
    return counts


def calculate_stats(results: Path) -> pd.DataFrame:
    counts = get_frame_counts(results)
    rows = []
    for exp in ALL_EXPERIMENTS:
        csv_file = results / f"{exp}_tracks.csv"
        if not csv_file.is_file():
            continue
        data = read_data(csv_file)
        durations = data.groupby("track_id")["frame"].nunique() if len(data) else pd.Series(dtype=int)
        det_frames = data.frame.nunique() if len(data) else 0
        multi_frames = int((data.groupby("frame").size() >= 2).sum()) if len(data) else 0
        total_frames = counts["internet" if exp.startswith("internet_") else "own"]
        rows.append({
            "tracker": exp,
            "frames": total_frames,
            "frames_with_tracks": int(det_frames),
            "coverage_pct": round(100 * det_frames / total_frames, 2) if total_frames else 0,
            "unique_ids": int(data.track_id.nunique()),
            "multi_object_frames": multi_frames,
            "multi_object_pct": round(100 * multi_frames / total_frames, 2) if total_frames else 0,
            "short_tracks_lt10": int((durations < 10).sum()),
            "median_track_frames": float(durations.median()) if len(durations) else 0.0,
            "max_track_frames": int(durations.max()) if len(durations) else 0,
            "detections": len(data),
        })
    return pd.DataFrame(rows)


def make_figures(stats: pd.DataFrame, results: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for prefix, selected, title in [
        ("own", stats[~stats.tracker.str.startswith("internet_")], "Собственное видео: 417 кадров"),
        ("internet", stats[stats.tracker.str.startswith("internet_")], "Видео из Интернета: 420 кадров"),
    ]:
        if selected.empty:
            continue
        for col, ylabel, suffix in [
            ("unique_ids", "Количество уникальных ID", "ids"),
            ("multi_object_frames", "Кадры с 2+ отслеживаемыми объектами", "multi"),
            ("median_track_frames", "Медианная длительность трека, кадры", "median"),
        ]:
            fig, ax = plt.subplots(figsize=(11.5, 5.0))
            ax.bar(selected.tracker, selected[col])
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", labelrotation=35)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            fig.tight_layout()
            fig.savefig(results / f"summary_{prefix}_{suffix}.png", dpi=175)
            plt.close(fig)


def analyze(results: Path) -> pd.DataFrame:
    results.mkdir(parents=True, exist_ok=True)
    df = calculate_stats(results)
    if df.empty:
        raise RuntimeError("Нет CSV с результатами трекинга в папке results")
    df.to_csv(results / "summary_all_experiments.csv", index=False, encoding="utf-8-sig")
    df[~df.tracker.str.startswith("internet_")].to_csv(results / "final_comparison_verified.csv", index=False, encoding="utf-8-sig")
    df[df.tracker.str.startswith("internet_")].to_csv(results / "internet_comparison_verified.csv", index=False, encoding="utf-8-sig")
    make_figures(df, results)
    print("\n=== ИТОГОВЫЙ АНАЛИЗ ФАКТИЧЕСКИХ CSV ===")
    print(df.to_string(index=False))
    print("\nСохранено в", results)
    return df


def ensure_configs(project: Path, results: Path) -> dict[str, Path]:
    import yaml
    configs = results / "tracker_configs"
    configs.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in BASES:
        dest = configs / (name + ".yaml")
        if not dest.is_file():
            bundled = project / "src" / "configs" / (name + ".yaml")
            if bundled.is_file():
                shutil.copy2(bundled, dest)
            else:
                from ultralytics.utils import ROOT
                shutil.copy2(ROOT / "cfg" / "trackers" / dest.name, dest)
        paths[name] = dest
        for mode, edits in MODES.items():
            tuned = configs / f"{name}_{mode}.yaml"
            if not tuned.is_file():
                with dest.open(encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                data.update(edits)
                tuned.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            paths[f"{name}_{mode}"] = tuned
    return paths


def video_info(path: Path) -> tuple[float, int, int]:
    import cv2
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    fps = video.get(cv2.CAP_PROP_FPS) or 20.0
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    if frames <= 0:
        raise RuntimeError(f"Видео без кадров: {path}")
    return fps, frames, 0


def prepare_internet(source: Path, target: Path) -> Path:
    if target.is_file():
        return target
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", "27",
        "-i", str(source), "-t", "28", "-vf", "fps=15,scale=720:-2", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-pix_fmt", "yuv420p", str(target),
    ], check=True)
    return target


def run_one(exp: str, video: Path, model_path: Path, config: Path, results: Path, cache: Path, device: str) -> None:
    import cv2
    from ultralytics import YOLO
    print("Обработка", exp)
    raw_path = cache / f"{exp}_raw.mp4"
    converted = cache / f"{exp}_result.mp4"
    csv_path = results / f"{exp}_tracks.csv"
    fps, _, _ = video_info(video)
    model = YOLO(str(model_path))
    outputs = model.track(source=str(video), tracker=str(config), imgsz=320,
                          conf=0.10, iou=0.50, device=device, stream=True,
                          persist=True, verbose=False)
    writer = None
    frame_count = 0
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            log = csv.writer(f)
            log.writerow(["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"])
            for res in outputs:
                frame_count += 1
                image = res.plot()
                h, w = image.shape[:2]
                if writer is None:
                    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Не удалось начать запись: {raw_path}")
                cv2.putText(image, exp.upper(), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                writer.write(image)
                boxes = res.boxes
                if boxes is None or boxes.id is None:
                    continue
                ids = boxes.id.int().cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                probs = boxes.conf.cpu().tolist()
                coords = boxes.xyxy.cpu().tolist()
                for tid, cls, prob, xyxy in zip(ids, classes, probs, coords):
                    log.writerow([frame_count, tid, res.names[cls], round(prob, 4),
                                  *[round(float(x), 2) for x in xyxy]])
    finally:
        if writer is not None:
            writer.release()
    if frame_count == 0:
        raise RuntimeError(f"Видеофайл не обработан: {video}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "26",
                    "-pix_fmt", "yuv420p", str(converted)], check=True)
    shutil.copy2(converted, results / converted.name)
    raw_path.unlink(missing_ok=True)
    print("Готово:", exp, frame_count, "кадров")


def run_all(base: Path, results: Path, project: Path, device: str, force: bool) -> None:
    model = base / "models" / "best.pt"
    videos = base / "videos"
    if not model.is_file():
        raise FileNotFoundError(f"Нужен файл обученной модели: {model}")
    internet = videos / "internet_rps.mp4"
    if not internet.is_file():
        raise FileNotFoundError(f"Нужно видео из Интернета: {internet}")
    candidates = sorted(p for p in videos.iterdir()
                        if p.suffix.lower() in {".mov", ".mp4", ".avi", ".mkv"}
                        and p.name != internet.name)
    if not candidates:
        raise FileNotFoundError(f"Не найдено собственное видео в {videos}")
    confs = ensure_configs(project, results)
    with tempfile.TemporaryDirectory(prefix="lr4_") as tmp:
        cache = Path(tmp)
        local_model = cache / "best.pt"
        shutil.copy2(model, local_model)
        own = cache / ("own" + candidates[0].suffix)
        shutil.copy2(candidates[0], own)
        internet_clip = prepare_internet(internet, cache / "internet_clip.mp4")
        own_frames = video_info(own)[1]
        int_frames = video_info(internet_clip)[1]
        (results / "metadata.json").write_text(json.dumps({
            "source_frames": {"own": own_frames, "internet": int_frames},
            "internet_clip": {"start_seconds": 27, "duration_seconds": 28, "fps": 15, "width": 720},
            "model": "YOLO26m (LR3, variant 18)",
            "input_model_location": str(model),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        specs = [(x, own, x) for x in FIRST_EXPERIMENTS]
        specs.extend([(x, internet_clip, x.removeprefix("internet_")) for x in SECOND_EXPERIMENTS])
        for exp, video, config_name in specs:
            if (results / f"{exp}_tracks.csv").is_file() and not force:
                print("Уже есть, пропуск:", exp)
                continue
            run_one(exp, video, local_model, confs[config_name], results, cache, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР4: YOLO26m + BoT-SORT/ByteTrack")
    parser.add_argument("--base", type=Path, default=Path("/content/drive/MyDrive/OIIS_LR4_Suhovilo"))
    parser.add_argument("--mode", choices=["analyze", "run", "all"], default="analyze")
    parser.add_argument("--device", default="0", help="0 — GPU Colab T4, cpu — без GPU")
    parser.add_argument("--force", action="store_true", help="Пересчитать и перезаписать результаты")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    results = args.base / "results"
    if args.mode in {"run", "all"}:
        run_all(args.base, results, project, args.device, args.force)
    if args.mode in {"analyze", "all"}:
        analyze(results)


if __name__ == "__main__":
    main()
