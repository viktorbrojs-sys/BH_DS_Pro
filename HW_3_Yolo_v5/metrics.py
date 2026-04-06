"""
Просмотр и визуализация метрик после обучения YOLOv5.
Показывает confusion matrix, кривые mAP, precision/recall, F1.
Копирует ключевые графики в outputs/plots/.

Запуск: python metrics.py [--run ПУТЬ_К_ЭКСПЕРИМЕНТУ]
"""

import os
import sys
import shutil
import argparse
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без GUI-окна, сохраняем в файл
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

warnings.filterwarnings("ignore")

# ─── Настройки ────────────────────────────────────────────────
RUNS_DIR   = Path("runs/train")
OUTPUT_DIR = Path("outputs/plots")
# ──────────────────────────────────────────────────────────────


def find_latest_run(runs_dir: Path) -> Path | None:
    """Ищет последний (по времени изменения) эксперимент в runs/train/."""
    if not runs_dir.exists():
        return None
    experiments = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not experiments:
        return None
    return max(experiments, key=lambda p: p.stat().st_mtime)


def parse_results_csv(run_dir: Path) -> dict | None:
    """Парсит results.csv из папки эксперимента YOLOv5."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None

    epochs, metrics = [], {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # Имена колонок могут содержать пробелы
    keys = {k.strip(): k for k in rows[0].keys()}

    def col(name_part):
        for k in keys:
            if name_part.lower() in k.lower():
                return keys[k]
        return None

    cols_needed = {
        "epoch":       col("epoch"),
        "box_loss":    col("box"),
        "obj_loss":    col("obj"),
        "cls_loss":    col("cls"),
        "precision":   col("precision"),
        "recall":      col("recall"),
        "mAP50":       col("map50") or col("map_0.5"),
        "mAP50_95":    col("map50-95") or col("map_0.5:0.95"),
    }

    result = {k: [] for k in cols_needed}
    for row in rows:
        for name, col_key in cols_needed.items():
            if col_key and col_key in row:
                try:
                    result[name].append(float(row[col_key]))
                except ValueError:
                    result[name].append(0.0)

    return result


def plot_training_curves(metrics: dict, save_path: Path):
    """Кривые потерь и метрик по эпохам."""
    epochs = list(range(1, len(metrics["mAP50"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Кривые обучения YOLOv5 — Детекция касок",
                 fontsize=14, fontweight="bold")

    # Потери (loss)
    ax = axes[0, 0]
    if metrics["box_loss"]:
        ax.plot(epochs, metrics["box_loss"], label="Box loss", color="#5599FF")
    if metrics["obj_loss"]:
        ax.plot(epochs, metrics["obj_loss"], label="Obj loss", color="#FF8844")
    if metrics["cls_loss"]:
        ax.plot(epochs, metrics["cls_loss"], label="Cls loss", color="#44BB44")
    ax.set_title("Потери (Loss)")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # mAP
    ax = axes[0, 1]
    if metrics["mAP50"]:
        ax.plot(epochs, metrics["mAP50"], label="mAP@0.5", color="#AA44FF", linewidth=2)
    if metrics["mAP50_95"]:
        ax.plot(epochs, metrics["mAP50_95"], label="mAP@0.5:0.95",
                color="#FF44AA", linewidth=2, linestyle="--")
    ax.set_title("mAP (Mean Average Precision)")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    # Precision
    ax = axes[1, 0]
    if metrics["precision"]:
        ax.plot(epochs, metrics["precision"], color="#44CCBB", linewidth=2)
    ax.set_title("Precision")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Recall
    ax = axes[1, 1]
    if metrics["recall"]:
        ax.plot(epochs, metrics["recall"], color="#FFBB33", linewidth=2)
    ax.set_title("Recall")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Кривые обучения сохранены: {save_path}")


def copy_yolo_plots(run_dir: Path, output_dir: Path):
    """Копирует стандартные графики YOLOv5 в outputs/plots/."""
    yolo_plots = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "results.png",
        "val_batch0_pred.jpg",
        "val_batch0_labels.jpg",
    ]

    copied = []
    for fname in yolo_plots:
        src = run_dir / fname
        if src.exists():
            dst = output_dir / fname
            shutil.copy2(src, dst)
            copied.append(fname)

    if copied:
        print(f"  [OK] Скопированы графики YOLOv5: {', '.join(copied)}")
    else:
        print("  [WARN] Стандартные графики YOLOv5 не найдены")


def display_yolo_image(img_path: Path, title: str, save_path: Path):
    """Показывает готовую картинку из YOLOv5 (confusion matrix и т.д.)."""
    if not img_path.exists():
        return
    img = mpimg.imread(str(img_path))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [OK] Сохранено: {save_path}")


def print_summary(metrics: dict, run_dir: Path):
    """Выводит итоговую сводку в консоль."""
    print("\n" + "=" * 55)
    print("  ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 55)

    if metrics and metrics["mAP50"]:
        best_map50 = max(metrics["mAP50"])
        best_epoch = metrics["mAP50"].index(best_map50) + 1
        last_prec  = metrics["precision"][-1] if metrics["precision"] else 0
        last_rec   = metrics["recall"][-1] if metrics["recall"] else 0
        last_map95 = metrics["mAP50_95"][-1] if metrics["mAP50_95"] else 0

        print(f"  Лучший mAP@0.5    : {best_map50:.4f} (эпоха {best_epoch})")
        print(f"  mAP@0.5:0.95      : {last_map95:.4f}")
        print(f"  Precision         : {last_prec:.4f}")
        print(f"  Recall            : {last_rec:.4f}")
        if last_prec + last_rec > 0:
            f1 = 2 * last_prec * last_rec / (last_prec + last_rec)
            print(f"  F1-Score          : {f1:.4f}")
    else:
        print("  Метрики не найдены (results.csv отсутствует)")

    print(f"\n  Папка эксперимента : {run_dir}")
    weights_best = run_dir / "weights" / "best.pt"
    if weights_best.exists():
        size_mb = weights_best.stat().st_size / 1024 / 1024
        print(f"  Лучшие веса        : {weights_best} ({size_mb:.1f} MB)")

    print("=" * 55)
    print("\n  Следующий шаг:")
    print("    python detect_rt.py --weights", weights_best)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Просмотр метрик после обучения YOLOv5"
    )
    parser.add_argument("--run", type=str, default=None,
                        help="Путь к папке эксперимента (по умолчанию: последний)")
    return parser.parse_args()


def main():
    print("=" * 55)
    print("  Метрики: Детекция касок (YOLOv5)")
    print("=" * 55)

    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Определяем папку эксперимента
    if args.run:
        run_dir = Path(args.run)
        if not run_dir.exists():
            print(f"[ОШИБКА] Папка не найдена: {run_dir}")
            sys.exit(1)
    else:
        run_dir = find_latest_run(RUNS_DIR)
        if run_dir is None:
            print("[ОШИБКА] Не найдено ни одного эксперимента в runs/train/")
            print("         Сначала запустите: python train.py")
            sys.exit(1)

    print(f"\n[INFO] Анализируем эксперимент: {run_dir.name}")

    # Парсим results.csv
    metrics = parse_results_csv(run_dir)
    if metrics and metrics["mAP50"]:
        print(f"  [OK] Найдено {len(metrics['mAP50'])} эпох в results.csv")
        plot_training_curves(
            metrics,
            OUTPUT_DIR / "training_curves.png"
        )
    else:
        print("  [WARN] results.csv не найден или пуст")

    # Копируем и перерисовываем стандартные графики YOLOv5
    print("\n[PLOTS] Копируем графики из эксперимента...")
    copy_yolo_plots(run_dir, OUTPUT_DIR)

    # Отдельно отображаем матрицу ошибок (если есть)
    cm_src = run_dir / "confusion_matrix.png"
    if cm_src.exists():
        display_yolo_image(
            cm_src,
            "Confusion Matrix — Детекция касок",
            OUTPUT_DIR / "confusion_matrix_view.png"
        )

    # Итоговая сводка
    print_summary(metrics, run_dir)
    print(f"\n  Все графики сохранены в: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
