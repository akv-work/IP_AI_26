import os
import urllib.request
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    dataset_path = 'dataset/data.yaml'

    if not os.path.exists(dataset_path):
        print("ОШИБКА: Датасет не найден!")
        print("Убедитесь, что папка 'dataset' с файлом data.yaml лежит рядом с lab3OIIS.py")
        exit()

    print("Инициализация модели YOLO11n...")
    model = YOLO('yolo11n.pt')

    print("Начало обучения (ускоренный режим)...")
    model.train(
        data=dataset_path,
        epochs=3,
        imgsz=320,
        batch=16,
        fraction=0.2,
        device='cpu'
    )

    print("Оценка эффективности на тестовой выборке (mAP)...")
    metrics = model.val(imgsz=320)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("Визуализация работы детектора...")
    url = "https://images.pexels.com/photos/1707820/pexels-photo-1707820.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    img_path = "test_image.jpg"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    success = False
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Изображение успешно загружено из интернета: {img_path}")
        success = True
    except Exception as e:
        print(f"Не удалось загрузить изображение из интернета: {e}")
        print("Использую локальное изображение из тестовой выборки...")

    if success:
        results = model(img_path, imgsz=320)
        for result in results:
            result.save(filename="result.jpg")
        img = Image.open("result.jpg")
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Результат обнаружения объектов')
        plt.show()
    else:
        test_images_dir = 'dataset/test/images'
        if os.path.exists(test_images_dir):
            image_files = os.listdir(test_images_dir)
            if image_files:
                local_img_path = os.path.join(test_images_dir, image_files[0])
                results = model(local_img_path, imgsz=320)
                for result in results:
                    result.save(filename="result.jpg")
                img = Image.open("result.jpg")
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title('Результат обнаружения (локальное изображение)')
                plt.show()
        else:
            print("Папка с тестовыми изображениями не найдена.")