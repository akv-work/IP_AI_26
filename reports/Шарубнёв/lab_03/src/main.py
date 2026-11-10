import os
import random
import glob
import torch
from ultralytics import YOLO


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство: {device}")

    data_yaml_path = 'vehicles-2/data.yaml'

    last_checkpoint = 'runs/detect/train/weights/last.pt'

    if os.path.exists(last_checkpoint):
        print("Найден предыдущий чекпоинт. Возобновляем обучение с места остановки...")
        model = YOLO(last_checkpoint)
        model.train(resume=True)
    else:
        print("Чекпоинт не найден. Начинаем обучение с нуля...")
        model = YOLO('yolo12s.pt')

        model.train(
            data=data_yaml_path,
            epochs=20,
            imgsz=416,
            batch=8,
            device=device,
            save=True,
            save_period=5,
            project='runs/detect',
            name='train_fast'
        )

    print("Запуск оценки модели на валидационной выборке...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")

    print("Тестирование работы детектора на локальном изображении...")
    test_images = glob.glob('vehicles-2/test/images/*.jpg')

    if test_images:
        test_image_path = random.choice(test_images)
        print(f"Выбрано изображение для теста: {test_image_path}")
        results = model.predict(source=test_image_path, device=device, save=True)
        print(f"Результат визуализации сохранен в папке: {results[0].save_dir}")
    else:
        print(
            "Тестовые изображения не найдены. Поместите любую картинку с машинами в папку проекта и укажите путь к ней.")


if __name__ == '__main__':
    main()