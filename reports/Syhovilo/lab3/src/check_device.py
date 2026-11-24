
import sys
import torch

print("=== ПРОВЕРКА КОМПЬЮТЕРА ===")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Видеокарта:", torch.cuda.get_device_name(0))
    print(
        "Видеопамять:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )
else:
    print("Обучение будет выполняться на CPU.")
