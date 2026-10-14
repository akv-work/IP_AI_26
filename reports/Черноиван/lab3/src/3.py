import argparse
from pathlib import Path

MODEL = "yolo12n.pt"
RUN_DIR = Path("runs/lab3")
RUN_NAME = "yolo12n_plates"


def download(api_key):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    ds = project.version(11).download("yolov11", location="datasets/plates")
    return Path(ds.location) / "data.yaml"


def fix_yaml(yaml_path):
    import yaml
    root = yaml_path.parent.resolve()
    cfg = yaml.safe_load(yaml_path.read_text())
    cfg["path"] = str(root)
    for k, d in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (root / d / "images").exists():
            cfg[k] = f"{d}/images"
        else:
            cfg.pop(k, None)
    yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    print("classes:", cfg.get("names"))
    return yaml_path


def train(args):
    from ultralytics import YOLO
    data = fix_yaml(download(args.api_key) if args.api_key else Path(args.data))
    model = YOLO(MODEL)
    model.train(data=str(data), epochs=args.epochs, imgsz=640, batch=args.batch,
                project=str(RUN_DIR), name=RUN_NAME, exist_ok=True, seed=0,
                device=args.device)
    best = RUN_DIR / RUN_NAME / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(data), split="test", project=str(RUN_DIR),
                            name=RUN_NAME + "_test", exist_ok=True, device=args.device)
    print(f"\nTEST  mAP50 = {m.box.map50:.4f}   mAP50-95 = {m.box.map:.4f}")
    print(f"      precision = {m.box.mp:.4f}   recall = {m.box.mr:.4f}")


def predict(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    res = model.predict(source=args.source, conf=args.conf, save=True,
                        project=str(RUN_DIR), name="predict", exist_ok=True)
    for r in res:
        print(Path(r.path).name, "->", len(r.boxes), "знак(ов)")
    print("Результаты:", RUN_DIR / "predict")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--api-key")
    t.add_argument("--data", default="datasets/plates/data.yaml")
    t.add_argument("--epochs", type=int, default=14)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--device", default=None)
    pr = sub.add_parser("predict")
    pr.add_argument("--source", required=True)
    pr.add_argument("--weights", default=str(RUN_DIR / RUN_NAME / "weights" / "best.pt"))
    pr.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()
    train(a) if a.cmd == "train" else predict(a)