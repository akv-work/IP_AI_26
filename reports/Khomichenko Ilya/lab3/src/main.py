from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import sys
import zipfile

BASE = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (BASE / path).resolve()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'prepare', 'train', 'evaluate', 'predict'], default='all')
    p.add_argument('--archive', help='Путь к ZIP, если рядом с main.py несколько архивов')
    p.add_argument('--dataset', help='Корень распакованного экспорта: train/, valid/ или val/, test/, data.yaml')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='auto', help='auto, cpu или 0 (первая NVIDIA GPU)')
    p.add_argument('--weights', help='Обученный best.pt для evaluate/predict; для train начальные веса')
    p.add_argument('--source', default='demo_images', help='Фото или папка для обнаружения людей')
    p.add_argument('--conf', type=float, default=0.25, help='Порог уверенности только для демонстрации')
    policy = p.add_mutually_exclusive_group()
    policy.add_argument('--skip-invalid', dest='skip_invalid', action='store_true',
                        help='Пропускать фото с ошибочной разметкой (включено по умолчанию)')
    policy.add_argument('--strict', dest='skip_invalid', action='store_false',
                        help='Остановиться при ошибках разметки вместо пропуска')
    p.set_defaults(skip_invalid=True)
    return p.parse_args()


def find_root(folder):
    candidates = [folder]
    candidates.extend(sorted({p.parent for p in folder.rglob('data.yaml')}))
    candidates.extend(sorted({p.parent for p in folder.rglob('train') if p.is_dir()}))
    roots = []
    for path in candidates:
        if ((path / 'train' / 'images').is_dir()
                and any((path / v / 'images').is_dir() for v in ['valid', 'val'])
                and (path / 'test' / 'images').is_dir() and path not in roots):
            roots.append(path)
    if len(roots) != 1:
        raise ValueError(f'Найдено корней датасета: {len(roots)}. Нужны train/images, '
                         'train/labels, valid/images (или val/images), valid/labels, '
                         'test/images, test/labels. Укажите точный корень через --dataset.')
    return roots[0]


def locate_dataset(args):
    if args.dataset:
        return find_root(local_path(args.dataset))
    unpacked = BASE / 'dataset'
    if unpacked.exists():
        return find_root(unpacked)
    archives = [local_path(args.archive)] if args.archive else sorted(BASE.glob('*.zip'))
    if len(archives) != 1:
        raise ValueError('Положите ZIP датасета рядом с main.py (один архив), '
                         'или задайте --archive "путь.zip", или --dataset "папка".')
    archive = archives[0]
    temporary = BASE / '_dataset_unpacking'
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        print(f'Распаковка: {archive.name}')
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target = (temporary / info.filename.replace('\\', '/')).resolve()
                if not target.is_relative_to(temporary.resolve()):
                    raise ValueError(f'Небезопасный путь в архиве: {info.filename}')
            z.extractall(temporary)
        # Не оставляем частично распакованный dataset при ошибке.
        find_root(temporary)
        temporary.rename(unpacked)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return find_root(unpacked)


def class_names(root, yaml):
    config_path = root / 'data.yaml'
    if not config_path.exists():
        raise ValueError(f'Нет {config_path}. Нужен data.yaml из экспорта Roboflow.')
    data = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('data.yaml должен содержать словарь.')
    names = data.get('names')
    if isinstance(names, dict):
        items = sorted((int(k), str(v)) for k, v in names.items())
        if [k for k, _ in items] != list(range(len(items))):
            raise ValueError('В names должны быть последовательные индексы от 0.')
        names = [v for _, v in items]
    if not isinstance(names, list) or len(names) != 1:
        raise ValueError(f'Для варианта 7 ожидается один класс людей. Получено names={names!r}.')
    if str(names[0]).strip().lower() not in {'person', 'people', 'human', 'humans', 'pedestrian', 'pedestrians'}:
        raise ValueError(f'Класс {names[0]!r} не похож на класс людей. Проверьте экспорт.')
    print(f'Классы из экспорта: {names}')
    return [str(names[0])]


def read_label(path, stats=None):
    """Принимает рамки YOLO и полигоны YOLO segmentation.
    Полигоны преобразуются в охватывающие рамки без изменения оригинала.
    """
    if not path.exists():
        raise ValueError('отсутствует файл разметки; для фонового фото нужен пустой .txt')
    rows, converted = [], 0
    eps = 1e-6
    for number, text in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        fields = text.split()
        if not fields:
            continue
        polygon = len(fields) >= 7 and (len(fields) - 1) % 2 == 0
        if len(fields) != 5 and not polygon:
            raise ValueError(f'строка {number}: ожидается 5 чисел для рамки или '
                             f'класс и не менее 3 пар координат для полигона; получено {len(fields)}')
        try:
            numbers = list(map(float, fields))
        except ValueError:
            raise ValueError(f'строка {number}: нечисловое значение') from None
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError(f'строка {number}: NaN или бесконечность')
        cls = numbers[0]
        if cls != 0:
            raise ValueError(f'строка {number}: для единственного класса ожидается индекс 0, получено {cls}')
        if polygon:
            coordinates = numbers[1:]
            if not all(-eps <= v <= 1 + eps for v in coordinates):
                raise ValueError(f'строка {number}: координаты полигона должны быть в [0,1]')
            coordinates = [min(1.0, max(0.0, v)) for v in coordinates]
            xs, ys = coordinates[0::2], coordinates[1::2]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
            w, h = xmax - xmin, ymax - ymin
            converted += 1
        else:
            x, y, w, h = numbers[1:]
        if not (-eps <= x <= 1 + eps and -eps <= y <= 1 + eps
                and 0 < w <= 1 + eps and 0 < h <= 1 + eps):
            raise ValueError(f'строка {number}: cx, cy должны быть в [0,1], w, h — в (0,1]; {text}')
        values = [min(1.0, max(0.0, v)) for v in [x, y, w, h]]
        # 10 десятичных знаков не должны превратить положительную рамку в нулевую.
        formatted = [f'{v:.10f}' for v in values]
        if float(formatted[2]) <= 0 or float(formatted[3]) <= 0:
            raise ValueError(f'строка {number}: вырожденная или слишком маленькая рамка')
        rows.append('0 ' + ' '.join(formatted))
    if stats is not None:
        stats['polygon_boxes'] = converted
    return '\n'.join(rows) + ('\n' if rows else '')


def prepare(args):
    import yaml
    from PIL import Image

    root = locate_dataset(args)
    names = class_names(root, yaml)
    reports = BASE / 'reports'
    reports.mkdir(exist_ok=True)
    staging = BASE / '_prepared_staging'
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    errors, counts = [], {}
    try:
        for split, source_split in [('train', 'train'), ('val', 'valid' if (root / 'valid').is_dir() else 'val'), ('test', 'test')]:
            source_images = root / source_split / 'images'
            source_labels = root / source_split / 'labels'
            if not source_labels.is_dir():
                raise ValueError(f'Нет папки {source_labels}')
            images = sorted(p for p in source_images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            stems = [p.stem.lower() for p in images]
            if len(stems) != len(set(stems)):
                raise ValueError(f'{source_images}: повторяющиеся имена без расширения; разметка неоднозначна.')
            target_images, target_labels = staging / split / 'images', staging / split / 'labels'
            target_images.mkdir(parents=True)
            target_labels.mkdir(parents=True)
            good = boxes = backgrounds = polygon_boxes = 0
            for img in images:
                label = source_labels / (img.stem + '.txt')
                try:
                    image_stats = {}
                    text = read_label(label, image_stats)
                    with Image.open(img) as picture:
                        picture.verify()
                except (ValueError, OSError, SyntaxError) as error:
                    errors.append({'split': split, 'image': str(img), 'label': str(label), 'reason': str(error)})
                    continue
                shutil.copy2(img, target_images / img.name)
                (target_labels / label.name).write_text(text, encoding='utf-8')
                good += 1
                polygon_boxes += image_stats['polygon_boxes']
                boxes += len(text.splitlines())
                backgrounds += not bool(text)
            counts[split] = {'images': good, 'boxes': boxes, 'backgrounds': backgrounds, 'polygon_boxes': polygon_boxes}
            print(f'{split}: {good} фото, {boxes} рамок, {backgrounds} фоновых фото; '
                  f'преобразовано полигонов: {polygon_boxes}')
        (reports / 'invalid_annotations.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
        (reports / 'dataset_summary.json').write_text(json.dumps({'source': str(root), 'names': names, 'splits': counts,
                                                                 'excluded_images': len(errors)}, ensure_ascii=False, indent=2), encoding='utf-8')
        if errors and not args.skip_invalid:
            first = errors[0]
            raise ValueError(f'Ошибок: {len(errors)}. Первая: {first["label"]}: {first["reason"]}. '
                             'Все причины в reports/invalid_annotations.json. Исправьте исходный экспорт '
                             'или явно разрешите исключение этих фотографий: --skip-invalid.')
        if errors:
            print(f'ВНИМАНИЕ: исключено {len(errors)} фото. Причины записаны в reports/invalid_annotations.json.')
        if any(v['images'] == 0 or v['boxes'] == 0 for v in counts.values()):
            raise ValueError('В каждом split нужны изображения и хотя бы одна размеченная рамка человека.')
        prepared = BASE / 'prepared_dataset'
        if prepared.exists():
            shutil.rmtree(prepared)
        staging.rename(prepared)
        config = {'path': prepared.as_posix(), 'train': 'train/images', 'val': 'val/images',
                  'test': 'test/images', 'nc': 1, 'names': names}
        path = prepared / 'data.yaml'
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        return path
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def device_for(args, torch):
    if args.device != 'auto':
        return args.device
    return 0 if torch.cuda.is_available() else 'cpu'


def saved_weights(args):
    if args.weights:
        result = local_path(args.weights)
    else:
        pointer = BASE / 'reports' / 'latest_run.json'
        if not pointer.exists():
            raise ValueError('Нет обученной модели. Сначала выполните обучение или укажите --weights "best.pt".')
        result = Path(json.loads(pointer.read_text(encoding='utf-8'))['best_weights'])
    if not result.is_file():
        raise ValueError(f'Не найдены веса: {result}')
    return result


def plot_history(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (directory / 'results.csv').open(encoding='utf-8') as file:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(file)]
    if not rows:
        return
    epoch = [float(r['epoch']) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key in ['train/box_loss', 'val/box_loss', 'train/cls_loss', 'val/cls_loss', 'train/dfl_loss', 'val/dfl_loss']:
        if key in rows[0]:
            axes[0].plot(epoch, [float(r[key]) for r in rows], label=key)
    for key in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
        if key in rows[0]:
            axes[1].plot(epoch, [float(r[key]) for r in rows], label=key)
    for axis, title in zip(axes, ['Training and validation losses', 'Validation mAP']):
        axis.set(title=title, xlabel='Epoch')
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(directory / 'learning_curves.png', dpi=160)
    plt.close(fig)


def train(args, data, YOLO, device):
    import torch
    import ultralytics
    initial = str(local_path(args.weights)) if args.weights else 'yolo12n.pt'
    print(f'Обучение YOLO12n: device={device}, epochs={args.epochs}, batch={args.batch}')
    model = YOLO(initial)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, workers=0, optimizer='AdamW', lr0=0.001,
                seed=42, deterministic=True, patience=15, amp=False,
                cache=False, plots=True, project=str(BASE / 'runs'), name='train', exist_ok=False)
    directory = Path(model.trainer.save_dir).resolve()
    best = directory / 'weights' / 'best.pt'
    if not best.exists():
        raise ValueError('Обучение не создало best.pt. Проверьте журнал обучения.')
    pointer = {'best_weights': str(best), 'run_directory': str(directory),
               'ultralytics': ultralytics.__version__, 'torch': torch.__version__}
    (BASE / 'reports' / 'latest_run.json').write_text(json.dumps(pointer, indent=2), encoding='utf-8')
    plot_history(directory)
    print(f'Лучшая модель: {best}')
    return best


def evaluate(args, data, weights, YOLO, device):
    model = YOLO(str(weights))
    metrics = model.val(data=str(data), split='test', imgsz=args.imgsz, batch=args.batch,
                        device=device, workers=0, plots=True,
                        project=str(BASE / 'runs'), name='test', exist_ok=False)
    values = {'model': 'YOLO12n', 'weights': str(weights), 'split': 'test',
              'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
              'mAP50': float(metrics.box.map50), 'mAP50-95': float(metrics.box.map)}
    folder = BASE / 'reports'
    folder.mkdir(exist_ok=True)
    (folder / 'test_metrics.json').write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    with (folder / 'test_metrics.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)
    print('\nРЕЗУЛЬТАТЫ НА TEST:')
    for key in ['precision', 'recall', 'mAP50', 'mAP50-95']:
        print(f'{key}: {values[key]:.4f} ({100 * values[key]:.2f}%)')


def predict(args, weights, YOLO, device):
    source = local_path(args.source)
    (BASE / 'demo_images').mkdir(exist_ok=True)
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        images = [source]
    elif source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    else:
        images = []
    if not images:
        print(f'Демонстрация: нет фото в {source}. Добавьте фотографии людей из Интернета '
              'в demo_images и выполните: python main.py --mode predict')
        return
    model = YOLO(str(weights))
    results = model.predict(source=[str(p) for p in images], imgsz=args.imgsz, conf=args.conf,
                            device=device, save=True, save_txt=True, save_conf=True,
                            project=str(BASE / 'runs'), name='predict', exist_ok=False)
    for item in results:
        count = len(item.boxes) if item.boxes is not None else 0
        print(f'{Path(item.path).name}: найдено людей — {count}; результат: {item.save_dir}')


def main():
    args = arguments()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.imgsz % 32 or not 0 <= args.conf <= 1:
        raise ValueError('epochs и batch >= 1; imgsz >= 32 и кратен 32; conf в [0,1].')
    (BASE / 'demo_images').mkdir(exist_ok=True)
    if args.mode in ['all', 'prepare', 'train']:
        data = prepare(args)
    else:
        data = BASE / 'prepared_dataset' / 'data.yaml'
    if args.mode == 'prepare':
        print(f'Датасет готов: {data}')
        return
    if args.mode == 'evaluate' and not data.exists():
        raise ValueError('Сначала выполните python main.py --mode prepare, чтобы подготовить test.')
    import torch
    from ultralytics import YOLO
    device = device_for(args, torch)
    print(f'PyTorch: {torch.__version__}; CUDA доступна: {torch.cuda.is_available()}; устройство: {device}')
    if device == 'cpu':
        print('Обучение на процессоре может занять много времени.')
    weights = train(args, data, YOLO, device) if args.mode in ['all', 'train'] else saved_weights(args)
    if args.mode in ['all', 'evaluate']:
        evaluate(args, data, weights, YOLO, device)
    if args.mode in ['all', 'predict']:
        predict(args, weights, YOLO, device)
    print('\nГотово. Результаты находятся в runs/ и reports/.')


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows / PyCharm
    try:
        main()
    except KeyboardInterrupt:
        print('\nВыполнение остановлено пользователем. Уже сохранённые результаты находятся в runs/.')
        sys.exit(130)
    except (ValueError, FileNotFoundError, zipfile.BadZipFile) as error:
        print(f'\nОШИБКА: {error}', file=sys.stderr)
        sys.exit(1)
    except ModuleNotFoundError as error:
        print(f'\nНе установлен модуль {error.name}. В терминале PyCharm: '
              'python -m pip install -r requirements.txt', file=sys.stderr)
        sys.exit(1)
