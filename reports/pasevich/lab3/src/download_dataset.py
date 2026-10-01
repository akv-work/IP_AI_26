import os
from huggingface_hub import snapshot_download

DATASET_DIR = "./vehicles-q0x2v"


def download_dataset():
    print("=" * 60)
    print("Загрузка датасета vehicles-q0x2v...")
    print("Это может занять 10-20 минут (~220 МБ)")
    print("=" * 60)

    snapshot_download(
        repo_id="LibreYOLO/vehicles-q0x2v",
        repo_type="dataset",
        local_dir=DATASET_DIR,
        local_dir_use_symlinks=False,
    )

    print(f"\nДатасет загружен: {os.path.abspath(DATASET_DIR)}")


def verify_dataset(dataset_path):
    print("\n--- Проверка структуры ---")

    for item in ["data.yaml", "train", "valid", "test"]:
        full = os.path.join(dataset_path, item)
        status = "OK" if os.path.exists(full) else "НЕ НАЙДЕН"
        print(f"  {status}: {item}")

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_path, split, "images")
        if os.path.isdir(img_dir):
            count = len([
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            print(f"  {split}: {count} изображений")


if __name__ == "__main__":
    download_dataset()
    verify_dataset(DATASET_DIR)
    print("\nГотово. Теперь запустите train.py")