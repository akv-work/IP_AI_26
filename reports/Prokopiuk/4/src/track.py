import argparse
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(BASE_DIR / "videos" / "rps2.mp4"))
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    ap.add_argument("--weights", default=str(BASE_DIR / "rsp12m_clean" / "weights" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.track(
        source=args.source,
        tracker=str(BASE_DIR / "trackers" / f"{args.tracker}.yaml"),
        imgsz=640,
        conf=args.conf,
        iou=0.5,
        save=True,
        project=BASE_DIR / "tracking_results",
        name=args.tracker,
        exist_ok=True,
    )
    print(f"сохранено: {BASE_DIR / 'tracking_results' / args.tracker}")


if __name__ == "__main__":
    main()
