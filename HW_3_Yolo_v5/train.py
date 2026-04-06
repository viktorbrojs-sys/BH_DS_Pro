"""
Обучение модели YOLOv5 для детекции касок на стройке.
Использует официальный скрипт train.py из репозитория YOLOv5.

Запуск: python train.py [--epochs N] [--batch N] [--weights MODEL] [--fraction F]

Быстрый запуск на CPU (≈3–5 мин на эпоху вместо 50):
  python train.py --fraction 0.1 --img 320 --epochs 10 --weights yolov5s.pt

  --fraction 0.1  — использовать только 10% датасета
  --img 320       — уменьшенный размер кадра (было 640, ускорение ~4x)
  --weights yolov5s.pt — лёгкая модель (было yolov5m)
"""

import os
import sys
import yaml
import shutil
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# ─── Настройки по умолчанию ───────────────────────────────────
YOLO_DIR    = Path("yolov5")
DATASET_DIR = Path("dataset")
DATA_YAML   = DATASET_DIR / "data.yaml"
RUNS_DIR    = Path("runs")
MINI_DIR    = Path("dataset_mini")   # куда кладём урезанный датасет

# Имя эксперимента — добавляем дату чтобы не перезаписывать
EXPERIMENT  = f"helmet_{datetime.now().strftime('%Y%m%d_%H%M')}"
# ──────────────────────────────────────────────────────────────


def check_prerequisites():
    """Проверяет наличие всего необходимого перед обучением."""
    errors = []

    if not YOLO_DIR.exists():
        errors.append(
            "Папка yolov5/ не найдена. Запустите сначала: python setup.py"
        )

    if not DATA_YAML.exists():
        errors.append(
            f"Файл {DATA_YAML} не найден. Поместите датасет в dataset/ "
            "и убедитесь, что data.yaml там есть."
        )

    if errors:
        print("[ОШИБКА] Не выполнены предварительные условия:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


def detect_device() -> str:
    """Определяет доступное устройство (GPU или CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  [GPU] Обнаружен: {gpu_name}")
            return "0"
        else:
            print("  [CPU] GPU не найден, будет использован CPU")
            return "cpu"
    except ImportError:
        print("  [CPU] PyTorch не установлен правильно, fallback на CPU")
        return "cpu"


def resolve_split_dir(data: dict, split: str, base: Path) -> Path | None:
    """Определяет абсолютный путь к папке с изображениями для сплита."""
    raw = data.get(split, "")
    if not raw:
        return None
    candidates = [
        Path(raw),
        base / raw,
        base / "images" / split,
        Path(raw.replace("../", "")),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def create_mini_dataset(original_yaml: Path, fraction: float, seed: int = 42) -> Path:
    """
    Копирует случайную долю (fraction) изображений и меток в dataset_mini/,
    создаёт новый data.yaml с путями к мини-датасету.

    fraction=0.1 → берём 10% файлов из train и val.
    Возвращает путь к новому data.yaml.
    """
    random.seed(seed)

    with open(original_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base = DATASET_DIR.resolve()
    mini_base = MINI_DIR.resolve()

    # Очищаем старый мини-датасет, чтобы не смешивать прогоны
    if mini_base.exists():
        shutil.rmtree(mini_base)

    new_data = dict(data)  # копируем nc, names и т.д.
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for split in ["train", "val", "test"]:
        src_img_dir = resolve_split_dir(data, split, base)
        if src_img_dir is None or not src_img_dir.exists():
            continue

        all_imgs = [p for p in src_img_dir.rglob("*")
                    if p.suffix.lower() in img_exts]

        if not all_imgs:
            continue

        # Количество файлов для сплита: минимум 1
        n_take = max(1, int(len(all_imgs) * fraction))
        chosen = random.sample(all_imgs, n_take)

        dst_img_dir = mini_base / "images" / split
        dst_lbl_dir = mini_base / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in chosen:
            # Копируем изображение
            shutil.copy2(img_path, dst_img_dir / img_path.name)

            # Копируем метку (если есть)
            lbl_path = Path(
                str(img_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
            ).with_suffix(".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

        new_data[split] = str(dst_img_dir)
        print(f"  [{split}] {len(all_imgs)} → взяли {n_take} файлов ({fraction*100:.0f}%)")

    # Сохраняем новый data.yaml
    mini_yaml = mini_base / "data.yaml"
    mini_base.mkdir(parents=True, exist_ok=True)
    with open(mini_yaml, "w", encoding="utf-8") as f:
        yaml.dump(new_data, f, allow_unicode=True, default_flow_style=False)

    print(f"  [OK] Мини-датасет создан: {mini_base}")
    return mini_yaml


def prepare_data_yaml(original: Path) -> Path:
    """
    Создаёт копию data.yaml с абсолютными путями к данным.
    YOLOv5 запускается из своей папки, поэтому относительные пути могут не работать.
    """
    with open(original, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    abs_dataset = DATASET_DIR.resolve()

    def abs_path(raw):
        if not raw:
            return raw
        p = Path(raw)
        if p.is_absolute() and p.exists():
            return str(p)
        for candidate in [abs_dataset / raw, abs_dataset / "images" / Path(raw).name, Path(raw.replace("../", ""))]:
            if candidate.exists():
                return str(candidate.resolve())
        return str((abs_dataset / raw).resolve())

    for split in ["train", "val", "test"]:
        if split in data:
            data[split] = abs_path(data[split])

    out_yaml = Path("data_train.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    print(f"  [INFO] Подготовлен data_train.yaml с абсолютными путями")
    return out_yaml


def build_train_command(args, device: str, data_yaml: Path) -> list:
    """Формирует команду запуска обучения YOLOv5."""
    cmd = [
        sys.executable,
        str(YOLO_DIR / "train.py"),
        "--img",     str(args.img),
        "--batch",   str(args.batch),
        "--epochs",  str(args.epochs),
        "--data",    str(data_yaml.resolve()),
        "--weights", args.weights,
        "--device",  device,
        "--name",    EXPERIMENT,
        "--project", str(RUNS_DIR.resolve() / "train"),
        "--exist-ok",  # не падать если папка уже есть
    ]

    # Дополнительные флаги
    if args.cache:
        cmd.append("--cache")

    if args.workers > 0:
        cmd += ["--workers", str(args.workers)]

    return cmd


def run_training(cmd: list):
    """Запускает обучение и выводит лог в реальном времени."""
    print("\n[TRAIN] Команда:")
    print("  " + " ".join(cmd))
    print("\n[TRAIN] Начало обучения... (Ctrl+C для остановки)\n")
    print("-" * 55)

    # Запускаем процесс без буферизации, чтобы видеть прогресс
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n[WARN] Обучение прервано пользователем")
        proc.terminate()
        proc.wait()
        return False

    proc.wait()
    return proc.returncode == 0


def print_results_hint(experiment: str):
    """Выводит подсказку где найти результаты."""
    results_dir = RUNS_DIR / "train" / experiment
    print("\n" + "=" * 55)
    if results_dir.exists():
        print(f"  [DONE] Результаты сохранены в: {results_dir}")
        print("\n  Что внутри:")
        print(f"    weights/best.pt    — лучшие веса")
        print(f"    weights/last.pt    — последние веса")
        print(f"    results.png        — кривые обучения")
        print(f"    confusion_matrix.png — матрица ошибок")
        print(f"    PR_curve.png       — precision-recall")
        print("\n  Следующие шаги:")
        print(f"    python metrics.py  — просмотр метрик")
        print(f"    python detect_rt.py — детекция в реальном времени")
    else:
        print(f"  [WARN] Папка результатов не найдена: {results_dir}")
        print("         Возможно обучение не завершилось корректно.")
    print("=" * 55)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение YOLOv5 для детекции касок",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--img",      type=int,   default=640,
                        help="Размер входного изображения (px)\n"
                             "На CPU рекомендуем 320 — ускорение ~4x")
    parser.add_argument("--batch",    type=int,   default=16,
                        help="Размер батча (-1 = авто)")
    parser.add_argument("--epochs",   type=int,   default=50,
                        help="Количество эпох обучения")
    parser.add_argument("--weights",  type=str,   default="yolov5m.pt",
                        help="Начальные веса: yolov5s/m/l/x.pt или scratch\n"
                             "На CPU рекомендуем yolov5s.pt — в 2x быстрее")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Доля датасета для обучения (0.0–1.0)\n"
                             "0.1 = 10%% файлов → эпоха в 10x быстрее\n"
                             "0.2 = 20%% — хороший баланс скорость/качество")
    parser.add_argument("--cache",    action="store_true",
                        help="Кэшировать датасет в RAM для ускорения")
    parser.add_argument("--workers",  type=int,   default=4,
                        help="Число потоков загрузки данных")
    return parser.parse_args()


def main():
    print("=" * 55)
    print("  Обучение: Детекция касок (YOLOv5)")
    print("=" * 55)

    args = parse_args()
    check_prerequisites()

    print("\n[CHECK] Определение устройства...")
    device = detect_device()

    # ── Авто-рекомендации для CPU ──────────────────────────────
    if device == "cpu":
        tips = []
        if args.batch > 4:
            args.batch = 4
            tips.append("batch → 4")
        if args.img > 320 and args.fraction < 1.0:
            # Если пользователь уже ограничил датасет, подскажем про img
            tips.append(f"совет: добавь --img 320 для доп. ускорения ~4x")
        if args.weights == "yolov5m.pt":
            # Подсказываем, но не меняем принудительно — пользователь решает сам
            tips.append("совет: --weights yolov5s.pt даст ускорение ~2x")
        if args.fraction == 1.0:
            tips.append(
                "совет: добавь --fraction 0.1 --img 320 --weights yolov5s.pt\n"
                "         → эпоха займёт ~3–5 мин вместо 50"
            )
        if tips:
            print("\n  [CPU СОВЕТЫ]")
            for t in tips:
                print(f"    • {t}")
    # ──────────────────────────────────────────────────────────

    # Валидация fraction
    if not (0.0 < args.fraction <= 1.0):
        print(f"[ОШИБКА] --fraction должен быть в диапазоне (0, 1], получено: {args.fraction}")
        sys.exit(1)

    print(f"\n[CONFIG]")
    print(f"  Изображение : {args.img}x{args.img}")
    print(f"  Батч        : {args.batch}")
    print(f"  Эпохи       : {args.epochs}")
    print(f"  Веса        : {args.weights}")
    print(f"  Устройство  : {device}")
    print(f"  Датасет     : {args.fraction*100:.0f}% от полного")
    print(f"  Эксперимент : {EXPERIMENT}")

    # ── Подготовка датасета ────────────────────────────────────
    if args.fraction < 1.0:
        print(f"\n[DATA] Создаём мини-датасет ({args.fraction*100:.0f}%)...")
        data_yaml = create_mini_dataset(DATA_YAML, args.fraction)
    else:
        print("\n[DATA] Подготовка путей к датасету...")
        data_yaml = prepare_data_yaml(DATA_YAML)
    # ──────────────────────────────────────────────────────────

    with open(data_yaml, "r") as f:
        d = yaml.safe_load(f)
    print(f"  train: {d.get('train', '???')}")
    print(f"  val  : {d.get('val', '???')}")
    print(f"  nc   : {d.get('nc', '???')}")
    print(f"  names: {d.get('names', '???')}")

    cmd = build_train_command(args, device, data_yaml)
    success = run_training(cmd)

    if success:
        print("\n[OK] Обучение успешно завершено!")
    else:
        print("\n[WARN] Обучение завершилось с ошибкой или было прервано.")

    print_results_hint(EXPERIMENT)

    # Удаляем временный yaml (только если полный датасет)
    if data_yaml.exists() and data_yaml.name == "data_train.yaml":
        data_yaml.unlink()


if __name__ == "__main__":
    main()
