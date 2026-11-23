
import os

# Ограничиваем число потоков, чтобы Windows оставался отзывчивым.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import sys
import json
import getpass
import traceback
from pathlib import Path
from urllib.request import Request, urlopen

try:
    import yaml
    import torch
    from PIL import Image
    from ultralytics import YOLO
except ImportError as error:
    print("\nНе найдена одна из необходимых библиотек:", error)
    print("\nОткрой терминал PyCharm в папке проекта и выполни:")
    print("python -m pip install -r requirements.txt")
    input("\nНажми Enter для выхода...")
    raise SystemExit(1)


# ---------------------------------------------------------------------
# НАСТРОЙКИ ЛАБОРАТОРНОЙ РАБОТЫ
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
RUNS_DIR = BASE_DIR / "runs"
RESULTS_DIR = BASE_DIR / "results"
EXTERNAL_IMAGES_DIR = BASE_DIR / "external_images"

MODEL_NAME = "yolo26m.pt"
RUN_NAME = "yolo26m_rock_paper_scissors_v14"

DATASET_WORKSPACE = "roboflow-58fyf"
DATASET_PROJECT = "rock-paper-scissors-sxsw"
DATASET_VERSION = 14

EPOCHS = 10
IMAGE_SIZE = 320
BATCH_SIZE = 4
DEVICE = "cpu"
WORKERS = 0
CONFIDENCE = 0.25

DATASET_PAGE = (
    "https://universe.roboflow.com/"
    "roboflow-58fyf/rock-paper-scissors-sxsw/dataset/14"
)

# Отдельные изображения жестов с Wikimedia Commons.
# Файлы распространяются по лицензии CC BY-SA 3.0.
INTERNET_IMAGES = {
    "rock.png": (
        "https://upload.wikimedia.org/wikipedia/commons/7/7e/"
        "Rock-paper-scissors_%28rock%29.png"
    ),
    "paper.png": (
        "https://upload.wikimedia.org/wikipedia/commons/a/af/"
        "Rock-paper-scissors_%28paper%29.png"
    ),
    "scissors.png": (
        "https://upload.wikimedia.org/wikipedia/commons/5/5f/"
        "Rock-paper-scissors_%28scissors%29.png"
    ),
}


# ---------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------

def create_project_folders():
    """Создаёт необходимые папки проекта."""
    for folder in (DATASET_DIR, RUNS_DIR, RESULTS_DIR, EXTERNAL_IMAGES_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def configure_cpu():
    """Настраивает PyTorch для работы на процессоре."""
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
    except RuntimeError:
        # PyTorch мог уже инициализировать пулы потоков.
        pass

    print("\nУстройство для обучения: CPU")
    print("Количество логических ядер:", os.cpu_count())
    print("Версия PyTorch:", torch.__version__)
    print("CUDA доступна:", torch.cuda.is_available())


def find_dataset_yaml():
    """Ищет YAML-файл датасета в папке dataset и её подпапках."""
    if not DATASET_DIR.exists():
        return None

    candidates = []
    candidates.extend(DATASET_DIR.rglob("data.yaml"))
    candidates.extend(DATASET_DIR.rglob("data.yml"))

    candidates = [
        path for path in candidates
        if path.is_file() and "runs" not in path.parts
    ]

    if not candidates:
        return None

    # Предпочитаем основной data.yaml в корне датасета.
    candidates.sort(key=lambda path: (len(path.parts), path.name.lower()))
    return candidates[0]


def download_dataset_from_roboflow():
    """
    Загружает именно вариант 14 датасета Roboflow
    в формате YOLO26.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("\nНе установлен пакет Roboflow.")
        print("Выполни: python -m pip install roboflow")
        return None

    print("\nДля загрузки публичного датасета Roboflow требуется API-ключ.")
    print("Ключ можно получить в настройках аккаунта:")
    print("https://app.roboflow.com/settings/api")
    print("Ключ вводится скрыто и не записывается в файлы проекта.")

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()

    if not api_key:
        try:
            api_key = getpass.getpass("\nВведи Roboflow API key: ").strip()
        except Exception:
            api_key = input("Введи Roboflow API key: ").strip()

    if not api_key:
        print("\nAPI-ключ не введён. Можно скачать датасет вручную.")
        print("Открой:", DATASET_PAGE)
        print("Выбери Download Dataset -> YOLO26.")
        print("Распакуй содержимое в папку:")
        print(DATASET_DIR)
        return None

    try:
        print("\nПодключение к Roboflow...")
        roboflow = Roboflow(api_key=api_key)
        project = roboflow.workspace(DATASET_WORKSPACE).project(DATASET_PROJECT)
        version = project.version(DATASET_VERSION)

        print("Загрузка датасета v14 в формате YOLO26...")
        dataset = version.download(
            "yolo26",
            location=str(DATASET_DIR),
            overwrite=False
        )

        print("\nЗагрузка завершена.")
        if hasattr(dataset, "location"):
            print("Папка датасета:", dataset.location)

        return find_dataset_yaml()

    except Exception as error:
        print("\nНе удалось загрузить датасет автоматически.")
        print("Причина:", error)
        print("\nМожно скачать датасет вручную:")
        print(DATASET_PAGE)
        print("Формат: YOLO26.")
        print("Распакуй файлы в папку dataset проекта.")
        return None


def prepare_dataset():
    """
    Проверяет датасет, при необходимости загружает его
    и создаёт локальный YAML с корректными путями.
    """
    create_project_folders()

    yaml_path = find_dataset_yaml()

    if yaml_path is None:
        print("\nДатасет не найден в папке:")
        print(DATASET_DIR)

        answer = input(
            "\nПопробовать загрузить датасет из Roboflow? (д/н): "
        ).strip().lower()

        if answer in ("д", "да", "y", "yes", ""):
            yaml_path = download_dataset_from_roboflow()

    if yaml_path is None:
        print("\nНе найден data.yaml.")
        print("Сначала скачай датасет v14 в формате YOLO26:")
        print(DATASET_PAGE)
        print("\nРаспакуй его в папку dataset и запусти программу снова.")
        return None

    print("\nНайден YAML датасета:", yaml_path)

    try:
        with open(yaml_path, "r", encoding="utf-8") as file:
            original_config = yaml.safe_load(file)
    except Exception as error:
        print("Не удалось прочитать YAML:", error)
        return None

    if not isinstance(original_config, dict):
        print("Ошибка: структура data.yaml некорректна.")
        return None

    names = original_config.get("names")

    if isinstance(names, dict):
        try:
            names = [
                names[key] for key in sorted(names, key=lambda item: int(item))
            ]
        except (ValueError, TypeError):
            names = list(names.values())

    if not isinstance(names, list) or len(names) != 3:
        print("Ошибка: в датасете ожидались три класса.")
        print("Найденные имена классов:", names)
        return None

    normalized_names = [str(name).strip().lower() for name in names]
    required_classes = {"rock", "paper", "scissors"}

    if set(normalized_names) != required_classes:
        print("Внимание: имена классов отличаются от ожидаемых.")
        print("Найдено:", names)
        print("Ожидаются классы: rock, paper, scissors.")
        return None

    # Ищем каталоги изображений по названиям split.
    split_aliases = {
        "train": {"train", "training"},
        "val": {"valid", "val", "validation"},
        "test": {"test", "testing"},
    }

    split_directories = {}

    all_image_dirs = [
        path for path in DATASET_DIR.rglob("images")
        if path.is_dir()
    ]

    for split_name, aliases in split_aliases.items():
        matching = [
            path for path in all_image_dirs
            if path.parent.name.lower() in aliases
            and any(
                file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                for file in path.iterdir()
                if file.is_file()
            )
        ]

        if not matching:
            print(f"Не найдена папка изображений для выборки: {split_name}")
            print("Проверь структуру датасета YOLO26.")
            return None

        split_directories[split_name] = matching[0]

    # Общая папка, содержащая train/images, valid/images и test/images.
    split_roots = [
        str(path.parent.parent.resolve())
        for path in split_directories.values()
    ]
    dataset_root = Path(os.path.commonpath(split_roots))

    normalized_config = {
        "path": str(dataset_root),
        "train": split_directories["train"].resolve().relative_to(
            dataset_root
        ).as_posix(),
        "val": split_directories["val"].resolve().relative_to(
            dataset_root
        ).as_posix(),
        "test": split_directories["test"].resolve().relative_to(
            dataset_root
        ).as_posix(),
        "nc": 3,
        "names": [str(name) for name in names],
    }

    # Проверяем наличие разметки YOLO рядом с изображениями.
    for split_name, images_dir in split_directories.items():
        labels_dir = images_dir.parent / "labels"

        if not labels_dir.exists():
            print(f"Предупреждение: папка labels не найдена для {split_name}:")
            print(labels_dir)
            return None

    local_yaml = BASE_DIR / "data_yolo26_local.yaml"

    try:
        with open(local_yaml, "w", encoding="utf-8") as file:
            yaml.safe_dump(
                normalized_config,
                file,
                allow_unicode=True,
                sort_keys=False
            )
    except Exception as error:
        print("Не удалось создать локальный YAML:", error)
        return None

    print("\nДатасет успешно проверен.")
    print("Классы:", ", ".join(names))
    print("Папка датасета:", dataset_root)

    for split_name, images_dir in split_directories.items():
        image_count = sum(
            1 for file in images_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        )
        print(f"{split_name}: {image_count} изображений")

    print("Конфигурация для обучения:", local_yaml)
    return local_yaml


def find_checkpoint(checkpoint_name):
    """Находит последний или лучший checkpoint."""
    expected = (
        RUNS_DIR / RUN_NAME / "weights" / checkpoint_name
    )

    if expected.exists():
        return expected

    candidates = list(RUNS_DIR.rglob(checkpoint_name))
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------
# ОБУЧЕНИЕ YOLO26m
# ---------------------------------------------------------------------

def train_model(resume=False):
    """Обучает YOLO26m или продолжает прерванное обучение."""
    data_yaml = prepare_dataset()

    if data_yaml is None:
        return

    configure_cpu()

    try:
        if resume:
            last_checkpoint = find_checkpoint("last.pt")

            if last_checkpoint is None:
                print("\nНе найден last.pt. Продолжать обучение нечего.")
                return

            print("\nПродолжение обучения из:", last_checkpoint)
            model = YOLO(str(last_checkpoint))

            model.train(
                data=str(data_yaml),
                resume=True,
                device=DEVICE,
                workers=WORKERS,
                batch=BATCH_SIZE,
                plots=True,
                amp=False,
                project=str(RUNS_DIR),
                name=RUN_NAME,
                exist_ok=True,
            )

        else:
            print("\nЗагрузка предобученной модели:", MODEL_NAME)
            model = YOLO(MODEL_NAME)

            print("\nНачинается обучение YOLO26m.")
            print("Эпох:", EPOCHS)
            print("Размер изображения:", IMAGE_SIZE)
            print("Batch:", BATCH_SIZE)
            print("Устройство:", DEVICE)
            print("\nВо время обучения не закрывай PyCharm и не выключай компьютер.")

            model.train(
                data=str(data_yaml),
                epochs=EPOCHS,
                imgsz=IMAGE_SIZE,
                batch=BATCH_SIZE,
                device=DEVICE,
                workers=WORKERS,
                optimizer="auto",
                patience=30,
                save=True,
                save_period=10,
                plots=True,
                cache=False,
                amp=False,
                val=True,
                project=str(RUNS_DIR),
                name=RUN_NAME,
                exist_ok=True,
                verbose=True,
                seed=42,
            )

        best_checkpoint = find_checkpoint("best.pt")
        last_checkpoint = find_checkpoint("last.pt")

        print("\nОбучение завершено.")
        if best_checkpoint:
            print("Лучшая модель:", best_checkpoint)
        if last_checkpoint:
            print("Последняя модель:", last_checkpoint)

        print("Графики и результаты обучения находятся в:")
        print(RUNS_DIR / RUN_NAME)

    except KeyboardInterrupt:
        print("\nОбучение остановлено пользователем.")
        print("Если была завершена хотя бы одна эпоха, попробуй пункт меню")
        print("'Продолжить прерванное обучение'.")

    except Exception as error:
        print("\nВо время обучения произошла ошибка:")
        print(error)
        traceback.print_exc()
        print("\nЕсли ошибка связана с нехваткой памяти, не меняя архитектуру,")
        print("можно уменьшить IMAGE_SIZE до 512 и повторить запуск.")


# ---------------------------------------------------------------------
# ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ
# ---------------------------------------------------------------------

def evaluate_test_set():
    """Оценивает лучшую модель на test split и сохраняет mAP."""
    data_yaml = prepare_dataset()

    if data_yaml is None:
        return

    best_checkpoint = find_checkpoint("best.pt")

    if best_checkpoint is None:
        print("\nНе найдена обученная модель best.pt.")
        print("Сначала выполни обучение.")
        return

    configure_cpu()

    try:
        print("\nЗагрузка лучшей модели:", best_checkpoint)
        model = YOLO(str(best_checkpoint))

        print("\nОценка качества на отдельной тестовой выборке...")
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            plots=True,
            project=str(RESULTS_DIR),
            name="test_evaluation",
            exist_ok=True,
            verbose=True,
        )

        box_metrics = metrics.box

        result_data = {
            "model": MODEL_NAME,
            "dataset": "Rock Paper Scissors v14",
            "test_mAP50_95": float(box_metrics.map),
            "test_mAP50": float(box_metrics.map50),
            "test_mAP75": float(box_metrics.map75),
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
            "per_class_mAP50_95": {},
        }

        class_names = model.names
        if isinstance(class_names, dict):
            class_names = [
                class_names[key] for key in sorted(class_names)
            ]

        per_class_maps = box_metrics.maps

        for index, class_name in enumerate(class_names):
            if index < len(per_class_maps):
                result_data["per_class_mAP50_95"][str(class_name)] = float(
                    per_class_maps[index]
                )

        metrics_file = RESULTS_DIR / "test_metrics.json"

        with open(metrics_file, "w", encoding="utf-8") as file:
            json.dump(
                result_data,
                file,
                ensure_ascii=False,
                indent=4
            )

        print("\n========== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ==========")
        print(f"mAP50-95: {result_data['test_mAP50_95']:.4f}")
        print(f"mAP50:    {result_data['test_mAP50']:.4f}")
        print(f"mAP75:    {result_data['test_mAP75']:.4f}")
        print(f"Precision:{result_data['precision']:.4f}")
        print(f"Recall:   {result_data['recall']:.4f}")

        print("\nТочность по классам (mAP50-95):")
        for class_name, value in result_data["per_class_mAP50_95"].items():
            print(f"{class_name}: {value:.4f}")

        print("\nМетрики сохранены:", metrics_file)
        print("Графики тестирования:", RESULTS_DIR / "test_evaluation")

    except Exception as error:
        print("\nОшибка при тестировании:", error)
        traceback.print_exc()


# ---------------------------------------------------------------------
# ИЗОБРАЖЕНИЯ ИЗ ИНТЕРНЕТА И ВИЗУАЛИЗАЦИЯ ДЕТЕКЦИИ
# ---------------------------------------------------------------------

def prepare_external_image(image_path):
    """
    Проверяет изображение и сохраняет его в JPG на белом фоне.
    Это корректно обрабатывает прозрачный фон PNG-файлов.
    """
    with Image.open(image_path) as image:
        image.load()
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)

        jpg_path = image_path.with_suffix(".jpg")
        background.convert("RGB").save(
            jpg_path,
            format="JPEG",
            quality=95
        )

    if image_path != jpg_path:
        image_path.unlink(missing_ok=True)

    return jpg_path


def download_internet_images():
    """Загружает три изображения жестов с Wikimedia Commons."""
    EXTERNAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        "Изображения жестов загружены с Wikimedia Commons.",
        "Автор: Sertion.",
        "Лицензия: Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0).",
        "Rock: https://commons.wikimedia.org/wiki/File:Rock-paper-scissors_(rock).png",
        "Paper: https://commons.wikimedia.org/wiki/File:Rock-paper-scissors_(paper).png",
        "Scissors: https://commons.wikimedia.org/wiki/File:Rock-paper-scissors_(scissors).png",
        "Изображения используются для демонстрации работы обученного детектора.",
        "Для обработки прозрачный фон PNG преобразуется в белый фон JPG.",
    ]

    downloaded = []

    for filename, url in INTERNET_IMAGES.items():
        image_path = EXTERNAL_IMAGES_DIR / filename
        jpg_path = image_path.with_suffix(".jpg")

        if jpg_path.exists():
            try:
                with Image.open(jpg_path) as image:
                    image.verify()
                downloaded.append(jpg_path)
                print("Изображение уже существует:", jpg_path.name)
                continue
            except Exception:
                jpg_path.unlink(missing_ok=True)

        if image_path.exists():
            try:
                converted_path = prepare_external_image(image_path)
                downloaded.append(converted_path)
                print("Подготовлено изображение:", converted_path.name)
                continue
            except Exception:
                image_path.unlink(missing_ok=True)

        try:
            print("\nЗагрузка изображения:", filename)
            request = Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Educational YOLO26 Lab)"}
            )

            with urlopen(request, timeout=60) as response:
                content = response.read()

            with open(image_path, "wb") as file:
                file.write(content)

            converted_path = prepare_external_image(image_path)
            downloaded.append(converted_path)
            print("Сохранено:", converted_path)

        except Exception as error:
            print("Не удалось загрузить", filename, ":", error)
            image_path.unlink(missing_ok=True)
            jpg_path.unlink(missing_ok=True)

    sources_file = EXTERNAL_IMAGES_DIR / "SOURCES.txt"
    with open(sources_file, "w", encoding="utf-8") as file:
        file.write("\n".join(sources))

    return downloaded


def predict_internet_images():
    """Запускает детектор на отдельных изображениях из Интернета."""
    best_checkpoint = find_checkpoint("best.pt")

    if best_checkpoint is None:
        print("\nНе найдена обученная модель best.pt.")
        print("Сначала выполни обучение.")
        return

    configure_cpu()

    images = download_internet_images()

    if not images:
        print("\nНе удалось автоматически загрузить изображения.")
        print("Положи JPG/PNG-фотографии жестов в папку:")
        print(EXTERNAL_IMAGES_DIR)
        print("Затем повтори этот пункт меню.")
        return

    try:
        model = YOLO(str(best_checkpoint))

        print("\nВыполняется распознавание изображений...")
        model.predict(
            source=[str(path) for path in images],
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(RESULTS_DIR),
            name="internet_predictions",
            exist_ok=True,
            verbose=True,
        )

        print("\nОбработка завершена.")
        print("Результаты с рамками и подписями находятся в:")
        print(RESULTS_DIR / "internet_predictions")
        print("\nТекстовые координаты обнаруженных объектов сохранены")
        print("в подпапке labels.")

    except Exception as error:
        print("\nОшибка при распознавании изображений:", error)
        traceback.print_exc()


# ---------------------------------------------------------------------
# ПОЛНЫЙ ЦИКЛ ЛАБОРАТОРНОЙ РАБОТЫ
# ---------------------------------------------------------------------

def run_full_pipeline():
    """Последовательно выполняет все основные этапы лабораторной."""
    print("\n========== ПОЛНЫЙ ЦИКЛ ЛР3 ==========")

    data_yaml = prepare_dataset()
    if data_yaml is None:
        return

    train_model(resume=False)

    best_checkpoint = find_checkpoint("best.pt")
    if best_checkpoint is None:
        print("\nОбученная модель не найдена. Тестирование пропущено.")
        return

    evaluate_test_set()
    predict_internet_images()

    print("\nОсновные этапы лабораторной работы завершены.")
    print("Проверь папки runs и results.")


# ---------------------------------------------------------------------
# ГЛАВНОЕ МЕНЮ
# ---------------------------------------------------------------------

def print_menu():
    print("\n" + "=" * 62)
    print("       ЛАБОРАТОРНАЯ РАБОТА № 3 — YOLO26m")
    print("       Вариант 18: Rock Paper Scissors v14")
    print("=" * 62)
    print("1. Подготовить или скачать датасет")
    print("2. Начать обучение YOLO26m")
    print("3. Продолжить прерванное обучение")
    print("4. Оценить модель на тестовой выборке (mAP)")
    print("5. Проверить отдельные изображения из Интернета")
    print("6. Выполнить полный цикл лабораторной работы")
    print("0. Выход")
    print("=" * 62)


def main():
    create_project_folders()
    configure_cpu()

    while True:
        print_menu()
        choice = input("Выбери действие: ").strip()

        if choice == "1":
            prepare_dataset()

        elif choice == "2":
            train_model(resume=False)

        elif choice == "3":
            train_model(resume=True)

        elif choice == "4":
            evaluate_test_set()

        elif choice == "5":
            predict_internet_images()

        elif choice == "6":
            run_full_pipeline()

        elif choice == "0":
            print("\nРабота программы завершена.")
            break

        else:
            print("\nНеизвестная команда. Выбери пункт от 0 до 6.")

        input("\nНажми Enter, чтобы вернуться в меню...")


if __name__ == "__main__":
    main()
