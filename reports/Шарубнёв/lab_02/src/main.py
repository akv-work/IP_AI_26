import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os


print("Загрузка датасета CIFAR-100...")
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
    'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

pretrained_model_filename = 'my_pretrained_densenet_lab2.h5'

if os.path.exists(pretrained_model_filename):
    print(f"Найдена сохраненная модель '{pretrained_model_filename}'. Загружаем веса...")
    model_pretrained = models.load_model(pretrained_model_filename)
else:
    print("Сохраненная модель не найдена. Начинаем построение и обучение с нуля...")

    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(72, 72, 3))
    base_model.trainable = False

    model_pretrained = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Resizing(72, 72),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dense(100, activation='softmax')
    ])

    model_pretrained.compile(optimizer=Adam(learning_rate=0.001),
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])


    history = model_pretrained.fit(x_train, y_train,
                                   epochs=10,
                                   batch_size=128,
                                   validation_data=(x_test, y_test))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Обучающая выборка')
    plt.plot(history.history['val_loss'], label='Тестовая выборка')
    plt.title('График ошибки (Loss)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Обучающая выборка')
    plt.plot(history.history['val_accuracy'], label='Тестовая выборка')
    plt.title('График точности (Accuracy)')
    plt.legend()
    plt.show()

    model_pretrained.save(pretrained_model_filename)
    print("Модель успешно сохранена!")

test_loss, test_acc = model_pretrained.evaluate(x_test, y_test, verbose=2)
print(f"Точность на тестовой выборке (Pre-trained DenseNet121): {test_acc:.4f}")



def predict_local_image(model, image_path, model_name="Модель"):
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл '{image_path}' не найден! Поместите картинку в папку со скриптом.")
        return

    try:
        img = Image.open(image_path).convert('RGB')

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis('off')

        img_resized = img.resize((32, 32))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        plt.title(
            f"{model_name}\nПредсказание: {cifar100_classes[predicted_class]}\nУверенность: {confidence * 100:.2f}%")
        plt.show()
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")


local_image_path = "img.png"

print("\n" + "=" * 50)
print("ВИЗУАЛИЗАЦИЯ РАБОТЫ ПРЕДОБУЧЕННОЙ СЕТИ")
print("=" * 50)
predict_local_image(model_pretrained, local_image_path, model_name="DenseNet121")

print("\n" + "=" * 50)
print("ВИЗУАЛИЗАЦИЯ РАБОТЫ КАСТОМНОЙ СЕТИ ИЗ ЛР №1")
print("=" * 50)

custom_model_filename = 'my_custom_model_lab1.h5'

if os.path.exists(custom_model_filename):
    try:
        custom_model = models.load_model(custom_model_filename)
        predict_local_image(custom_model, local_image_path, model_name="Кастомная сеть (ЛР №1)")
    except Exception as e:
        print(f"Ошибка при загрузке кастомной модели: {e}")
else:
    print(f"Файл '{custom_model_filename}' не найден. Создаю базовую необученную сеть...")

    custom_model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(100, activation='softmax')
    ])

    predict_local_image(custom_model, local_image_path, model_name="Кастомная сеть (Без весов)")