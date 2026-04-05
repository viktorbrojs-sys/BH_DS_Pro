"""
Скрипт установки окружения для детекции касок на стройке.
Клонирует репозиторий YOLOv5, устанавливает зависимости, проверяет окружение.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path


def run_cmd(cmd, description=""):
    """Запуск команды с выводом результата."""
    print(f"\n>>> {description or cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"[ОШИБКА] {result.stderr}")
        return False
    return True


def check_python():
    """Проверка версии Python."""
    version = sys.version_info
    print(f"Python версия: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ОШИБКА] Требуется Python 3.8 или выше")
        sys.exit(1)
    print("[OK] Python версия подходит")


def clone_yolov5():
    """Клонирование официального репозитория YOLOv5."""
    yolo_dir = Path("yolov5")

    if yolo_dir.exists():
        print("[INFO] Папка yolov5 уже существует, пропускаем клонирование")
        return True

    print("\n[SETUP] Клонирование YOLOv5 из официального репозитория...")
    ok = run_cmd(
        "git clone https://github.com/ultralytics/yolov5.git",
        "Клонирование github.com/ultralytics/yolov5"
    )
    if not ok:
        print("[ОШИБКА] Не удалось клонировать YOLOv5. Проверьте интернет-соединение.")
        return False

    print("[OK] YOLOv5 успешно клонирован")
    return True


def install_yolov5_requirements():
    """Установка зависимостей YOLOv5."""
    req_file = Path("./yolov5/requirements.txt")
    if not req_file.exists():
        print("[ОШИБКА] Файл ./yolov5/requirements.txt не найден")
        return False

    print("\n[SETUP] Установка зависимостей YOLOv5...")
    ok = run_cmd(
        f"{sys.executable} -m pip install -r ./yolov5/requirements.txt",
        "Установка зависимостей из yolov5/requirements.txt"
    )
    return ok


def install_extra_requirements():
    """Установка дополнительных зависимостей проекта."""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("[WARN] Файл requirements.txt не найден, пропускаем")
        return True

    print("\n[SETUP] Установка дополнительных зависимостей...")
    ok = run_cmd(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Установка зависимостей из requirements.txt"
    )
    return ok


def check_torch():
    """Проверка установки PyTorch и доступности GPU."""
    print("\n[CHECK] Проверка PyTorch и устройств...")
    try:
        import torch
        print(f"  PyTorch версия: {torch.__version__}")

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  GPU доступен: {device_name}")
            print(f"  CUDA версия: {torch.version.cuda}")
            print("  [OK] Обучение будет на GPU (device=0)")
        else:
            print("  GPU не обнаружен — будет использован CPU")
            print("  [WARN] Обучение на CPU медленнее, рекомендуем уменьшить --epochs")

        return True
    except ImportError:
        print("  [ОШИБКА] PyTorch не установлен")
        return False


def check_opencv():
    """Проверка установки OpenCV."""
    try:
        import cv2
        print(f"  OpenCV версия: {cv2.__version__}")
        print("  [OK] OpenCV установлен")
        return True
    except ImportError:
        print("  [ОШИБКА] OpenCV не установлен")
        return False


def setup_dataset_dir():
    """Создание структуры папок для датасета."""
    print("\n[SETUP] Подготовка структуры датасета...")

    dirs = [
        "dataset",
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images/test",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test",
        "outputs",
        "outputs/plots",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("  Структура папок создана:")
    for d in dirs:
        print(f"    {d}/")

    # Создаём инструкцию по датасету
    readme = Path("dataset/README_dataset.txt")
    if not readme.exists():
        readme.write_text(
            "=== Инструкция по подготовке датасета Hard Hat Workers ===\n\n"
            "Датасет: https://universe.roboflow.com/yolo-whioc/hard-hat-workers-laqdp\n\n"
            "Шаги:\n"
            "1. Скачайте ZIP с Roboflow в формате 'YOLOv5 PyTorch'\n"
            "2. Распакуйте архив в эту папку (dataset/)\n"
            "3. Убедитесь, что файл data.yaml находится в dataset/data.yaml\n"
            "4. В data.yaml должны быть пути:\n"
            "     train: ../dataset/images/train\n"
            "     val:   ../dataset/images/val\n"
            "     test:  ../dataset/images/test  (если есть)\n\n"
            "Классы датасета:\n"
            "  0: head       - голова без каски (нарушение!)\n"
            "  1: helmet     - каска (норма)\n\n"
            "После распаковки запустите: python eda.py\n",
            encoding="utf-8"
        )

    print("  [OK] Инструкция сохранена в dataset/README_dataset.txt")


def check_dataset():
    """Проверка наличия датасета."""
    yaml_path = Path("dataset/data.yaml")
    if yaml_path.exists():
        print("  [OK] Датасет найден: dataset/data.yaml")
    else:
        print("  [WARN] Датасет не найден. Поместите распакованный датасет в папку dataset/")
        print("         Читайте: dataset/README_dataset.txt")


def main():
    print("=" * 60)
    print("  Установка окружения: Детекция касок (YOLOv5)")
    print("=" * 60)

    check_python()

    ok = clone_yolov5()
    if not ok:
        sys.exit(1)

    install_yolov5_requirements()
    install_extra_requirements()

    print("\n[CHECK] Проверка установленных компонентов...")
    check_torch()
    check_opencv()

    setup_dataset_dir()
    check_dataset()

    print("\n" + "=" * 60)
    print("  Установка завершена!")
    print("=" * 60)
    print("\nДальнейшие шаги:")
    print("  1. Поместите датасет в папку dataset/ (см. dataset/README_dataset.txt)")
    print("  2. python eda.py        — анализ датасета")
    print("  3. python train.py      — обучение модели")
    print("  4. python metrics.py    — просмотр метрик")
    print("  5. python detect_rt.py  — детекция в реальном времени")


if __name__ == "__main__":
    main()
