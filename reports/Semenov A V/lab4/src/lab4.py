
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import urllib.request
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
VIDEOS = [
    dict(name="orange_kittens", url="https://upload.wikimedia.org/wikipedia/commons/transcoded/7/72/Three_orange_kittens_following_their_mother.webm/Three_orange_kittens_following_their_mother.webm.480p.vp9.webm",
         page="https://commons.wikimedia.org/wiki/File:Three_orange_kittens_following_their_mother.webm",
         author="Jeromi Mikhael", license="CC0 1.0", start=0),
    dict(name="cat_and_kittens", url="https://upload.wikimedia.org/wikipedia/commons/4/47/Cat_and_kittens.webm",
         page="https://commons.wikimedia.org/wiki/File:Cat_and_kittens.webm",
         author="Catfan", license="CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/", start=63),
]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dependencies():
    try:
        global cv2, np, torch, yaml, YOLO, ultralytics
        import cv2
        import numpy as np
        import torch
        import yaml
        import ultralytics
        import lap
        from ultralytics import YOLO
        import matplotlib
    except ImportError as e:
        raise RuntimeError("Установите зависимости: python -m pip install ultralytics==8.4.14 lap==0.5.12 matplotlib") from e


def download(url, target):
    target = Path(target)
    if target.is_file() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    error = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Lab4Educational/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, partial.open("wb") as f:
                expected = int(r.headers.get("Content-Length", "0"))
                received = 0
                while chunk := r.read(1024 * 1024):
                    f.write(chunk)
                    received += len(chunk)
                if expected and received != expected:
                    raise OSError(f"Неполная загрузка: {received}/{expected} байт")
            partial.replace(target)
            return target
        except (OSError, ValueError) as e:
            error = e
            partial.unlink(missing_ok=True)
            print(f"Загрузка: повтор {attempt + 1}/3: {e}", flush=True)
    raise RuntimeError(f"Не удалось скачать {url}. Скачайте видео и передайте --source путь. {error}")


def find_weights(value):
    if value:
        p = Path(value).expanduser().resolve()
    elif (BASE / "best.pt").is_file():
        p = BASE / "best.pt"
    else:
        candidates = sorted((BASE / "results_lab3").glob("*/train/weights/best.pt"))
        if not candidates:
            raise FileNotFoundError("Нужны обученные веса ЛР 3: положите best.pt рядом с lab4.py или задайте --weights.")
        p = candidates[-1]
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def configurations():
    result = {}
    folder = Path(ultralytics.__file__).parent / "cfg/trackers"
    for tracker in ("bytetrack", "botsort"):
        base = yaml.safe_load((folder / f"{tracker}.yaml").read_text(encoding="utf-8"))
        # Явные значения делают сравнение воспроизводимым в данной версии.
        base.update(track_high_thresh=.25, track_low_thresh=.1, new_track_thresh=.25,
                    track_buffer=30, match_thresh=.8, fuse_score=True)
        if tracker == "botsort":
            base.update(gmc_method="sparseOptFlow", with_reid=False)
        result[f"{tracker}_base"] = base.copy()
        for label, changes in {
            "buffer5": {"track_buffer": 5}, "buffer90": {"track_buffer": 90},
            "match50": {"match_thresh": .5}, "high60": {"track_high_thresh": .6},
        }.items():
            result[f"{tracker}_{label}"] = {**base, **changes}
        if tracker == "botsort":
            result["botsort_no_gmc"] = {**base, "gmc_method": "none"}
    return result


def validate_config(c):
    for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        if not isinstance(c[key], (float, int)) or not 0 <= c[key] <= 1:
            raise ValueError(f"{key} должен быть числом от 0 до 1")
    if c["track_low_thresh"] >= c["track_high_thresh"]:
        raise ValueError("track_low_thresh должен быть меньше track_high_thresh")
    if type(c["track_buffer"]) is not int or c["track_buffer"] < 0:
        raise ValueError("track_buffer должен быть целым неотрицательным")


def track_video(weights, source, config_path, out, args, start_seconds=0):
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Видео не открывается: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Не удалось определить FPS: {source}")
    frame_limit = round(args.seconds * fps) if args.seconds > 0 else None
    start_frame = round(start_seconds * fps)
    if start_frame:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            cap.release()
            raise RuntimeError("Не удалось перейти к началу фрагмента")
    # Новый экземпляр сбрасывает Kalman, список потерянных треков и ID.
    model = YOLO(str(weights))
    if model.task != "detect" or list(model.names.values()) != ["cat"]:
        cap.release()
        raise ValueError("Нужны веса детектора одного класса cat из ЛР 3")
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    histories = defaultdict(lambda: deque(maxlen=60))
    seen_frames = defaultdict(list)
    counts, timings = [], []
    writer = None
    frame_no = 0
    multi_snapshot_saved = False
    snapshot_step = max(1, round(3 * fps))
    start = time.perf_counter()
    try:
        with (out / "tracks.csv").open("w", newline="", encoding="utf-8") as f:
            log = csv.writer(f)
            log.writerow(["frame", "time_s", "id", "class", "confidence", "x1", "y1", "x2", "y2", "source_frame", "source_time_s"])
            while frame_limit is None or frame_no < frame_limit:
                ok, frame = cap.read()
                if not ok:
                    break
                # Сохраняем пропорции; обработка всех кадров без пропуска.
                if frame.shape[1] > args.width:
                    height = round(frame.shape[0] * args.width / frame.shape[1] / 2) * 2
                    frame = cv2.resize(frame, (args.width, height))
                h, w = frame.shape[:2]
                if writer is None:
                    writer = cv2.VideoWriter(str(out / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Не удалось создать выходное MP4")
                t = time.perf_counter()
                r = model.track(frame, persist=True, tracker=str(config_path), imgsz=args.imgsz,
                                conf=args.conf, device=device, verbose=False)[0]
                timings.append(time.perf_counter() - t)
                boxes = r.boxes
                active = []
                if boxes is not None and boxes.id is not None:
                    for xyxy, score, cls, identity in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                                                        boxes.cls.cpu().tolist(), boxes.id.int().cpu().tolist()):
                        x1, y1, x2, y2 = xyxy
                        active.append(identity)
                        seen_frames[identity].append(frame_no)
                        histories[identity].append((int((x1+x2)/2), int((y1+y2)/2)))
                        log.writerow([frame_no, frame_no/fps, identity, int(cls), score, *xyxy, start_frame+frame_no, (start_frame+frame_no)/fps])
                        color = ((identity*73)%200+55, (identity*151)%200+55, (identity*41)%200+55)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"cat ID:{identity} {score:.2f}", (max(0,int(x1)), max(20,int(y1)-5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
                        cv2.polylines(frame, [np.array(histories[identity], np.int32)], False, color, 2)
                counts.append(len(active))
                cv2.putText(frame, f"{config_path.stem}  frame={frame_no}  tracks={len(active)}", (10, h-12),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
                writer.write(frame)
                if len(active) >= 2 and not multi_snapshot_saved:
                    cv2.imencode(".jpg", frame)[1].tofile(str(out / "first_multiple.jpg"))
                    multi_snapshot_saved = True
                if frame_no % snapshot_step == 0:
                    # imencode поддерживает пути Windows с кириллицей.
                    cv2.imencode(".jpg", frame)[1].tofile(str(out / f"frame_{frame_no:06d}.jpg"))
                frame_no += 1
                if args.show:
                    cv2.imshow("Lab4 - press Q to stop current run", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    if frame_no == 0:
        raise ValueError("Видео не содержит читаемых кадров")
    gaps = sum(sum(b > a+1 for a,b in zip(fs, fs[1:])) for fs in seen_frames.values())
    summary = dict(video=source.stem, config=config_path.stem, frames=frame_no, fps=fps,
                   source_start_frame=start_frame, source_start_seconds=start_frame/fps,
                   seconds=frame_no/fps, unique_ids=len(seen_frames), observations=sum(counts),
                   mean_active=float(np.mean(counts)), zero_frames=counts.count(0),
                   multi_frames=sum(n >= 2 for n in counts), return_gaps=gaps,
                   mean_track_observations=float(np.mean([len(v) for v in seen_frames.values()])) if seen_frames else 0,
                   processing_seconds=time.perf_counter()-start,
                   track_fps_warm=(len(timings)-1)/sum(timings[1:]) if len(timings)>1 else None)
    dump(out / "summary.json", summary)
    dump(out / "frame_counts.json", counts)
    dump(out / "id_frames.json", seen_frames)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def make_plots(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    videos = list(dict.fromkeys(r["video"] for r in rows))
    fig, axes = plt.subplots(len(videos), 2, figsize=(10, 4.7*len(videos)), squeeze=False)
    for i, video in enumerate(videos):
        subset = [r for r in rows if r["video"] == video]
        for j, (key, title) in enumerate((("unique_ids", "Число выданных ID"), ("mean_active", "Среднее число активных треков"))):
            labels = [r["config"].replace("bytetrack_", "Byte ").replace("botsort_", "BoT ") for r in subset]
            axes[i,j].barh(labels, [r[key] for r in subset])
            axes[i,j].set_title(video + "\n" + title, fontsize=12)
            axes[i,j].tick_params(labelsize=11)
            axes[i,j].grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", help="best.pt из ЛР 3")
    p.add_argument("--source", nargs="+", help="Локальные видео или прямые HTTP(S)-ссылки на видео")
    p.add_argument("--configs", nargs="+", help="Имена конфигураций; по умолчанию все 11")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Изменить параметр YAML во всех выбранных конфигурациях")
    p.add_argument("--seconds", type=float, default=15, help="Первые N секунд, 0 = всё видео")
    p.add_argument("--start", type=float, help="Начало в секундах; по умолчанию A=0, B=63; для --source=0")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--width", type=int, default=854, help="Максимальная ширина видео, чётное число")
    p.add_argument("--conf", type=float, default=.05, help="Порог детектора; ниже track_low_thresh")
    p.add_argument("--device", help="cpu или индекс CUDA, например 0")
    p.add_argument("--show", action="store_true")
    p.add_argument("--out", type=Path, default=BASE / "results_lab4")
    args = p.parse_args()
    if args.start is not None and (not math.isfinite(args.start) or args.start < 0):
        p.error("--start должен быть конечным неотрицательным числом")
    if not math.isfinite(args.seconds) or args.seconds < 0 or args.imgsz < 32 or args.width < 32 or args.width % 2 or not 0 < args.conf < 1:
        p.error("Некорректные seconds, imgsz, width или conf")
    dependencies()
    weights = find_weights(args.weights)
    configs = configurations()
    if args.configs:
        unknown = set(args.configs) - configs.keys()
        if unknown:
            p.error(f"Неизвестные конфигурации {unknown}; доступны {list(configs)}")
        configs = {k: configs[k] for k in dict.fromkeys(args.configs)}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or key == "tracker_type" or any(key not in c for c in configs.values()):
            p.error(f"Недопустимый общий параметр {item}")
        for c in configs.values():
            c[key] = yaml.safe_load(value)
    for c in configs.values():
        validate_config(c)
        if args.conf > c["track_low_thresh"]:
            print("Внимание: conf выше track_low_thresh, часть слабых детекций отсекается до трекера.")
    out = args.out.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    config_dir = out / "configs"
    config_dir.mkdir()
    sources, metadata, starts = [], [], []
    if args.source:
        for i, s in enumerate(args.source):
            if s.startswith(("https://", "http://")):
                from urllib.parse import urlsplit
                suffix = Path(urlsplit(s).path).suffix or ".mp4"
                src = download(s, BASE / "data/lab4" / (hashlib.sha256(s.encode()).hexdigest()[:12]+suffix))
            else:
                src = Path(s).expanduser().resolve()
            if not src.is_file():
                raise FileNotFoundError(src)
            sources.append(src)
            starts.append(args.start or 0)
            metadata.append(dict(source=s, path=str(src), sha256=sha256(src)))
    else:
        for v in VIDEOS:
            src = download(v["url"], BASE / "data/lab4" / (v["name"] + ".webm"))
            sources.append(src)
            starts.append(v["start"] if args.start is None else args.start)
            metadata.append({**v, "path":str(src), "sha256":sha256(src)})
    dump(out / "manifest.json", dict(python=platform.python_version(), torch=torch.__version__,
         ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         weights=str(weights), weights_sha256=sha256(weights), sources=metadata,
         arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
         notes="ID counts and gaps are descriptive statistics, NOT ID switches, MOTA or IDF1. Videos are silent annotated derivatives; retain source attribution/license."))
    rows = []
    for name, c in configs.items():
        path = config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
        for i, source in enumerate(sources):
            rows.append(track_video(weights, source, path, out / f"{i+1}_{source.stem}" / name, args, starts[i]))
            dump(out / "summary.json", rows)
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    make_plots(rows, out)
    print(f"Готово. Результаты: {out}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
