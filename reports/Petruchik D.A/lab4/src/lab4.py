import os
import time
import random
import shutil
import subprocess
from pathlib import Path
from getpass import getpass
from collections import defaultdict

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import ultralytics

from roboflow import Roboflow
from ultralytics import YOLO
from google.colab import files
from IPython.display import Video, display

SEED = 42
EPOCHS = 20
IMGSZ = 640
BATCH = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

START = 0
DURATION = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path("/content/lab4_cats")
ROOT.mkdir(exist_ok=True)

RUN = ROOT / time.strftime("run_%Y%m%d_%H%M%S")
RUN.mkdir(exist_ok=False)

WEIGHTS = ROOT / "cats_yolo12s_best.pt"

print("Вариант 10: YOLO12s, коты")
print("Устройство:", DEVICE)
print("Результаты:", RUN)

if not WEIGHTS.exists():
    api_key = os.getenv("ROBOFLOW_API_KEY") or getpass(
        "Введите API-ключ Roboflow: "
    )

    rf = Roboflow(api_key=api_key)

    dataset = (
        rf.workspace("mohamed-traore-2ekkp")
        .project("cats-n9b87")
        .version(3)
        .download(
            "yolov8",
            location=str(ROOT / "dataset")
        )
    )

    dataset_dir = Path(dataset.location).resolve()
    original_yaml = dataset_dir / "data.yaml"

    data_cfg = yaml.safe_load(
        original_yaml.read_text(encoding="utf-8")
    )

    data_cfg["path"] = str(dataset_dir)

    for key, split in [
        ("train", "train"),
        ("val", "valid"),
        ("test", "test"),
    ]:
        folder = dataset_dir / split / "images"

        if not folder.exists() or not any(
            p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            for p in folder.iterdir()
        ):
            raise RuntimeError(f"Нет изображений: {folder}")

        data_cfg[key] = str(folder)

    yaml_path = RUN / "data.yaml"

    yaml_path.write_text(
        yaml.safe_dump(data_cfg, allow_unicode=True),
        encoding="utf-8",
    )

    print("Классы датасета:", data_cfg["names"])

    model = YOLO("yolo12s.pt")

    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=0,
        seed=SEED,
        project=str(RUN),
        name="training",
        plots=True,
    )

    trained_weights = (
        Path(model.trainer.save_dir) / "weights" / "best.pt"
    )

    shutil.copy2(trained_weights, WEIGHTS)

    metrics = YOLO(str(WEIGHTS)).val(
        data=str(yaml_path),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=0,
        project=str(RUN),
        name="test_evaluation",
    )

    detector_metrics = pd.DataFrame([{
        "Precision": float(metrics.box.mp),
        "Recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }])

    detector_metrics.to_csv(
        RUN / "detector_metrics.csv",
        index=False,
    )

    display(detector_metrics)

else:
    print("Используются ранее обученные веса:", WEIGHTS)

check_model = YOLO(str(WEIGHTS))

names = check_model.names
items = names.items() if isinstance(names, dict) else enumerate(names)

CAT_IDS = [
    int(class_id)
    for class_id, name in items
    if str(name).strip().lower()
    in {"cat", "cats", "кот", "коты", "кошка", "кошки"}
]

if not CAT_IDS:
    raise RuntimeError(
        f"Не найден класс котов. Имена классов модели: {names}"
    )

del check_model

print("ID классов котов:", CAT_IDS)

print("Загрузи один ролик с несколькими котами.")
uploaded = files.upload()

if len(uploaded) != 1:
    raise RuntimeError("Нужно загрузить ровно один видеофайл.")

uploaded_name = next(iter(uploaded))

source = Path(uploaded_name).resolve()

source_url = input(
    "Ссылка на источник ролика для отчёта "
    "(если видео своё — напиши «собственное»): "
).strip()

(RUN / "video_source.txt").write_text(
    f"Файл: {uploaded_name}\n"
    f"Источник: {source_url or 'Не указан'}\n"
    f"Начало: {START} с\n"
    f"Длительность фрагмента: {DURATION} с\n",
    encoding="utf-8",
)

VIDEO = RUN / "video.mp4"

subprocess.run(
    [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(source),
        "-ss", str(START),
        "-t", str(DURATION),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(VIDEO),
    ],
    check=True,
)

cap = cv2.VideoCapture(str(VIDEO))
ok, _ = cap.read()

video_fps = cap.get(cv2.CAP_PROP_FPS)
video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.release()

if not ok:
    raise RuntimeError("Видео не читается или выбран пустой фрагмент.")

print(
    f"Кадров: {video_frames}, FPS: {video_fps:.2f}, "
    f"размер: {video_width} × {video_height}"
)

TRACKERS_DIR = (
    Path(ultralytics.__file__).resolve().parent
    / "cfg" / "trackers"
)

CONFIGS = RUN / "trackers"
CONFIGS.mkdir()

for name in ("botsort.yaml", "bytetrack.yaml"):
    cfg = yaml.safe_load(
        (TRACKERS_DIR / name).read_text(encoding="utf-8")
    )

    cfg.update(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )

    if name == "botsort.yaml":
        cfg.update(
            gmc_method="sparseOptFlow",
            with_reid=False,
        )

    (CONFIGS / name).write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )


def make_cfg(base_yaml, out_name, changes):
    cfg = yaml.safe_load(
        Path(base_yaml).read_text(encoding="utf-8")
    )
    cfg.update(changes)

    out = CONFIGS / out_name
    out.write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    return out

CONF = 0.1

RESULTS = []
ALL_TRACKS = {}
OUTPUT_DIRS = {}


def run_tracking(tracker_cfg, label, folder, conf=CONF):
    model = YOLO(str(WEIGHTS))

    tracks = defaultdict(list)
    records = []

    n_frames = 0
    n_dets = 0
    max_objects = 0
    frames_with_tracks = 0

    started = time.perf_counter()

    stream = model.track(
        source=str(VIDEO),
        tracker=str(tracker_cfg),
        classes=CAT_IDS,
        conf=conf,
        imgsz=IMGSZ,
        device=DEVICE,
        stream=True,
        save=True,
        project=str(RUN / "tracking"),
        name=folder,
        exist_ok=False,
        verbose=False,
    )

    for frame_number, result in enumerate(stream):
        n_frames += 1

        boxes = result.boxes

        if boxes is None or boxes.id is None:
            continue

        ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        n_dets += len(ids)
        max_objects = max(max_objects, len(ids))
        frames_with_tracks += bool(ids)

        for track_id, xyxy, confidence in zip(
            ids, coordinates, confidences
        ):
            tracks[track_id].append(frame_number)

            records.append({
                "frame": frame_number,
                "time_s": frame_number / video_fps,
                "track_id": track_id,
                "confidence": confidence,
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3],
            })

    elapsed = time.perf_counter() - started

    if n_frames == 0:
        raise RuntimeError(f"Не обработаны кадры: {label}")

    lengths = np.array(
        [len(frames) for frames in tracks.values()],
        dtype=int,
    )

    output_dir = Path(model.predictor.save_dir)
    OUTPUT_DIRS[label] = output_dir

    pd.DataFrame(
        records,
        columns=[
            "frame", "time_s", "track_id", "confidence",
            "x1", "y1", "x2", "y2",
        ],
    ).to_csv(output_dir / "tracks.csv", index=False)

    stats = {
        "эксперимент": label,
        "кадров": n_frames,
        "уникальных ID": len(tracks),
        "средняя длина трека (кадров)": (
            round(float(lengths.mean()), 2)
            if lengths.size else 0.0
        ),
        "коротких треков (<15 кадров)": int(
            (lengths < 15).sum()
        ),
        "объектов на кадр": round(n_dets / n_frames, 3),
        "максимум одновременно": max_objects,
        "доля кадров с треками": round(
            frames_with_tracks / n_frames, 3
        ),
        "FPS полного прохода": round(
            n_frames / max(elapsed, 1e-9), 2
        ),
    }

    return stats, dict(tracks)

BYTE = CONFIGS / "bytetrack.yaml"
BOT = CONFIGS / "botsort.yaml"

EXPERIMENTS = [
    ("BoT-SORT: база", "bot_base", BOT, {}, CONF),
    ("ByteTrack: база", "byte_base", BYTE, {}, CONF),

    ("ByteTrack: buffer=10", "byte_buf10",
     BYTE, {"track_buffer": 10}, CONF),

    ("ByteTrack: buffer=90", "byte_buf90",
     BYTE, {"track_buffer": 90}, CONF),

    ("ByteTrack: high=0.5", "byte_high05",
     BYTE, {"track_high_thresh": 0.5}, CONF),

    ("ByteTrack: match=0.5", "byte_match05",
     BYTE, {"match_thresh": 0.5}, CONF),

    ("ByteTrack: match=0.95", "byte_match095",
     BYTE, {"match_thresh": 0.95}, CONF),

    ("ByteTrack: conf=0.25", "byte_conf025",
     BYTE, {}, 0.25),

    ("BoT-SORT: buffer=10", "bot_buf10",
     BOT, {"track_buffer": 10}, CONF),

    ("BoT-SORT: buffer=90", "bot_buf90",
     BOT, {"track_buffer": 90}, CONF),

    ("BoT-SORT: без GMC", "bot_nogmc",
     BOT, {"gmc_method": "none"}, CONF),

    ("BoT-SORT: conf=0.25", "bot_conf025",
     BOT, {}, 0.25),
]

for label, folder, base, changes, conf in EXPERIMENTS:
    config_path = make_cfg(
        base, f"{folder}.yaml", changes
    )

    print("Запуск:", label)

    stats, tracks = run_tracking(
        config_path, label, folder, conf
    )

    RESULTS.append(stats)
    ALL_TRACKS[label] = tracks

    pd.DataFrame(RESULTS).to_csv(
        RUN / "comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Готово:", label)

df = pd.DataFrame(RESULTS)
display(df)

columns = [
    "уникальных ID",
    "средняя длина трека (кадров)",
    "коротких треков (<15 кадров)",
]

fig, axes = plt.subplots(
    1, 3, figsize=(19, 8), sharey=True
)

for ax, column in zip(axes, columns):
    ax.barh(df["эксперимент"], df[column])
    ax.set_title(column)
    ax.grid(alpha=0.3, axis="x")

axes[0].invert_yaxis()

plt.tight_layout()
plt.savefig(RUN / "comparison.png", dpi=160)
plt.show()

labels = ["BoT-SORT: база", "ByteTrack: база"]

fig, axes = plt.subplots(
    1, 2, figsize=(16, 7), sharex=True
)

for ax, label in zip(axes, labels):
    items = sorted(
        ALL_TRACKS[label].items(),
        key=lambda item: item[1][0],
    )

    for row, (track_id, frames) in enumerate(items):

        ax.scatter(
            frames,
            np.full(len(frames), row),
            marker="|",
            s=12,
        )

    ax.set_title(f"{label}\nКоличество ID: {len(items)}")
    ax.set_xlabel("Номер кадра")
    ax.set_ylabel("Трек по порядку появления")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RUN / "track_visibility.png", dpi=160)
plt.show()

def show_tracked(label):
    folder = OUTPUT_DIRS[label]

    videos = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in {".avi", ".mp4"}
        and not p.name.endswith("_preview.mp4")
    )

    if not videos:
        raise FileNotFoundError(
            f"Нет выходного видео в {folder}"
        )

    output = folder / "result_preview.mp4"

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(videos[0]),
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "26",
            str(output),
        ],
        check=True,
    )

    display(Video(str(output), embed=True, width=800))


show_tracked("ByteTrack: база")
show_tracked("BoT-SORT: база")


report = [
    "# Лабораторная работа №4",
    "",
    "## Вариант 10",
    "Детектор YOLO12s. Объекты: коты.",
    "",
    "## Цель",
    "Исследовать отслеживание нескольких объектов "
    "алгоритмами ByteTrack и BoT-SORT.",
    "",
    "## Подготовка",
    "Детектор обучается внутри этой работы на датасете "
    "cats-n9b87 версии 3. При повторном запуске в той же "
    "среде используются сохранённые веса.",
    f"Эпох при новом обучении: {EPOCHS}.",
    f"Размер входа: {IMGSZ}. Устройство: {DEVICE}.",
    f"Источник видео: {source_url or 'Не указан'}.",
    f"Обработано кадров фрагмента: {video_frames}.",
    "",
    "## Методика",
    "Все эксперименты используют одинаковые веса и видео. "
    "Для каждого эксперимента создаётся новая модель. "
    "В каждом опыте изменяется один параметр относительно "
    "базовой конфигурации соответствующего алгоритма.",
    "",
    "## Результаты",
    "| Эксперимент | ID | Средняя длина | Короткие треки | FPS |",
    "|---|---:|---:|---:|---:|",
]

for row in RESULTS:
    report.append(
        f"| {row['эксперимент']} "
        f"| {row['уникальных ID']} "
        f"| {row['средняя длина трека (кадров)']} "
        f"| {row['коротких треков (<15 кадров)']} "
        f"| {row['FPS полного прохода']} |"
    )

report += [
    "",
    "![Сравнение](comparison.png)",
    "",
    "![Видимость треков](track_visibility.png)",
    "",
    "## Интерпретация",
    "track_buffer определяет время сохранения потерянного "
    "трека. Увеличение помогает переживать перекрытия, но "
    "может повысить риск ошибочного сопоставления.",
    "",
    "track_high_thresh задаёт порог сильных детекций "
    "первого этапа. При повышении порога их становится меньше.",
    "",
    "Уменьшение match_thresh делает ограничение по стоимости "
    "сопоставления строже.",
    "",
    "gmc_method управляет компенсацией движения камеры. "
    "В базовом BoT-SORT включён sparseOptFlow; ReID выключен.",
    "",
    "Повышение conf до 0.25 отсекает слабые детекции до "
    "передачи трекеру и ограничивает второй этап ByteTrack.",
    "",
    "## Ограничения",
    "Уникальные ID не равны числу котов. "
    "Меньшее число ID само по себе не доказывает лучшее качество.",
    "Средняя длина — количество кадров с наблюдением ID, "
    "а не интервал между первым и последним появлением.",
    "Без эталонной покадровой разметки не вычислялись "
    "IDF1, MOTA и число смен идентичности.",
    "FPS включает обработку, отрисовку, запись видео и прогрев.",
    "",
    "## Визуальный анализ перед сдачей",
    "Просмотрите сохранённые видео и добавьте наблюдения "
    "с номерами кадров: смены ID, пропуски, ложные рамки "
    "и сохранение ID при перекрытии котов.",
]

if df["максимум одновременно"].max() < 2:
    report += [
        "",
        "ВНИМАНИЕ: одновременно не обнаружены два трека. "
        "Для демонстрации множественного трекинга требуется "
        "другой ролик или улучшение детектора.",
    ]

(RUN / "REPORT.md").write_text(
    "\n".join(report),
    encoding="utf-8",
)

(RUN / "versions.txt").write_text(
    f"ultralytics={ultralytics.__version__}\n"
    f"torch={torch.__version__}\n",
    encoding="utf-8",
)

shutil.copy2(WEIGHTS, RUN / "cats_yolo12s_best.pt")

archive = shutil.make_archive(
    str(RUN),
    "zip",
    root_dir=str(RUN),
)

print("Работа завершена.")
print("Отчёт:", RUN / "REPORT.md")
print("Архив:", archive)