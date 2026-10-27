import os
import sys
from ultralytics import YOLO


def train_model(data_yaml_path):
    print("=== [1] НАЧАЛО БЫСТРОГО ОБУЧЕНИЯ (YOLO11n) на CPU с нуля ===")
    # Загружаем предобученную нано-модель YOLO11
    model = YOLO("yolo11n.pt")

    # Обучаем модель
    results = model.train(
        data=data_yaml_path,
        epochs=4,
        imgsz=416,
        batch=4,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        seed=16,
        device='cpu',
        project="results_lab3_var2",
        name="train",
        cache=False
    )

    # ВАЖНО: Получаем путь к лучшим весам напрямую из результатов обучения
    # Это защищает от ошибки, если Ultralytics создаст папку train-5, train-6 и т.д.
    best_weights = str(results.save_dir / "weights" / "best.pt")
    print(f"=== ОБУЧЕНИЕ ЗАВЕРШЕНО. Веса сохранены: {best_weights} ===")
    return best_weights


def evaluate_model(data_yaml_path, weights_path):
    print("\n=== [2] ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ===")
    model = YOLO(weights_path)

    # Валидация на тестовом сплите
    metrics = model.val(data=data_yaml_path, split="test", imgsz=416, device='cpu')

    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")


def predict_images(weights_path, source_path):
    print(f"\n=== [3] ВИЗУАЛИЗАЦИЯ (Источник: {source_path}) ===")
    model = YOLO(weights_path)

    # Предсказание с порогом уверенности 0.25
    results = model.predict(
        source=source_path,
        conf=0.25,
        save=True,  # Сохранить картинки с рамками
        project="results_lab3_var2",
        name="predict",
        device='cpu'
    )
    print("=== РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ СОХРАНЕНЫ ===")
    print("Проверьте папку results_lab3_var2/predict")


if __name__ == "__main__":
    # Путь к вашему файлу data.yaml (он лежит рядом с этим скриптом)
    DATA_YAML = "data.yaml"

    # Путь к картинке с машиной (убедитесь, что она лежит в этой же папке)
    TEST_IMAGE_SOURCE = "car.jpg"

    if not os.path.exists(DATA_YAML):
        print(f"ОШИБКА: Файл {DATA_YAML} не найден в папке {os.getcwd()}!")
        sys.exit(1)

    # 1. ОБУЧЕНИЕ (Запускается с нуля)
    weights = train_model(DATA_YAML)

    # 2. Оценка на тесте
    evaluate_model(DATA_YAML, weights)

    # 3. Визуализация (если указан путь)
    if os.path.exists(TEST_IMAGE_SOURCE):
        predict_images(weights, TEST_IMAGE_SOURCE)
    else:
        print(f"\n[Пропуск визуализации] Файл {TEST_IMAGE_SOURCE} не найден.")
        print(
            "Чтобы протестировать детектор, скачайте фото машины с номером, назовите его 'car.jpg' и положите в папку с проектом.") 