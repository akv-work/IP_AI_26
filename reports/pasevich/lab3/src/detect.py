import os
import sys
import cv2
import numpy as np
import requests
from ultralytics import YOLO

WEIGHTS = "runs/detect/vehicles_yolo12s/weights/best.pt"
OUTPUT_DIR = "detection_results"
CONF_THRESHOLD = 0.2
DEVICE = "cpu"


def load_model(weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Веса не найдены: {weights_path}\n"
            f"Сначала запустите train.py и дождитесь завершения обучения."
        )
    print(f"Загрузка модели: {weights_path}")
    return YOLO(weights_path)


def download_image(url):
    print(f"Скачивание изображения: {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Не удалось декодировать изображение")

    return img


def detect_and_save(model, image, source_name):
    print("Выполняется детекция...")
    results = model(image, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
    result = results[0]

    annotated = result.plot()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{source_name}_result.jpg")
    cv2.imwrite(out_path, annotated)

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        print(f"\nОбнаружено объектов: {len(boxes)}")
        names = result.names
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  [{names[cls_id]}] "
                  f"уверенность={conf:.3f} "
                  f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    else:
        print("\nОбъекты не обнаружены. Попробуйте уменьшить CONF_THRESHOLD.")

    print(f"\nРезультат сохранён: {out_path}")
    return result


def detect_from_url(model, url):
    img = download_image(url)
    name = url.split("/")[-1].split("?")[0]
    name = os.path.splitext(name)[0][:30] or "web_image"
    return detect_and_save(model, img, name)


def detect_from_file(model, path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Не удалось прочитать файл: {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    return detect_and_save(model, img, name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python detect.py <URL изображения>")
        print("  python detect.py --file <путь к файлу>")
        print()
        print("Пример для PyCharm:")
        print("  Run → Edit Configurations → Parameters:")
        print("  https://example.com/traffic.jpg")
        sys.exit(1)

    model = load_model(WEIGHTS)

    if sys.argv[1] == "--file":
        if len(sys.argv) < 3:
            print("Укажите путь к файлу после --file")
            sys.exit(1)
        detect_from_file(model, sys.argv[2])
    else:
        detect_from_url(model, sys.argv[1])