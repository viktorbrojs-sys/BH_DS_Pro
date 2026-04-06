"""
Разведочный анализ датасета Hard Hat Workers.
Визуализирует примеры изображений с bounding boxes, распределение классов.
"""

import os
import sys
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Пробуем подключить OpenCV
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    print("[WARN] OpenCV не найден, используем Pillow для загрузки изображений")
    from PIL import Image

# ─── Настройки ────────────────────────────────────────────────
DATASET_DIR   = Path("dataset")
DATA_YAML     = DATASET_DIR / "data.yaml"
OUTPUT_DIR    = Path("outputs/plots")
SAMPLES_GRID  = 12   # сколько картинок показать в сетке
RANDOM_SEED   = 42

# Цвета для классов (BGR → RGB для matplotlib)
CLASS_COLORS = {
    0: "#FF4444",   # head — красный (нарушение)
    1: "#44BB44",   # helmet — зелёный (норма)
}
# ──────────────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict:
    """Загружает data.yaml с описанием датасета."""
    if not path.exists():
        print(f"[ОШИБКА] Файл не найден: {path}")
        print("         Поместите датасет в папку dataset/ и запустите снова.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_img_dir(data: dict, split: str) -> Path:
    """
    Разбирает пути из data.yaml — Roboflow может указывать абсолютные
    или относительные пути, пробуем несколько вариантов.
    """
    raw = data.get(split, "")
    if not raw:
        return None

    candidates = [
        Path(raw),
        DATASET_DIR / raw,
        DATASET_DIR / "images" / split,
        Path(raw.replace("../", "")),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_images(img_dir: Path, limit: int = 500) -> list:
    """Собирает список изображений из папки (jpg/jpeg/png)."""
    if img_dir is None or not img_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    return imgs[:limit]


def load_image(path: Path):
    """Загружает изображение в формате RGB numpy array."""
    if HAVE_CV2:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = Image.open(path).convert("RGB")
        return np.array(img)


def load_labels(img_path: Path) -> list:
    """
    Загружает YOLO-метки для изображения.
    Формат: class_id cx cy w h (нормализованные координаты)
    """
    # Папка labels обычно находится рядом с images
    label_path = Path(str(img_path).replace("/images/", "/labels/").replace(
        "\\images\\", "\\labels\\"))
    label_path = label_path.with_suffix(".txt")

    if not label_path.exists():
        return []

    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                labels.append((cls_id, cx, cy, w, h))
    return labels


def draw_boxes(ax, img, labels: list, class_names: list):
    """Рисует bounding boxes на изображении."""
    ax.imshow(img)
    h_img, w_img = img.shape[:2]

    for cls_id, cx, cy, bw, bh in labels:
        # Конвертируем из нормализованных координат в пиксели
        x1 = int((cx - bw / 2) * w_img)
        y1 = int((cy - bh / 2) * h_img)
        box_w = int(bw * w_img)
        box_h = int(bh * h_img)

        color = CLASS_COLORS.get(cls_id, "#FFFF00")
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        rect = patches.Rectangle(
            (x1, y1), box_w, box_h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4, cls_name,
            color="white", fontsize=7, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none")
        )

    ax.axis("off")


def plot_sample_grid(images: list, class_names: list, split: str, save_path: Path):
    """Сетка случайных изображений с bounding boxes."""
    random.seed(RANDOM_SEED)
    sample = random.sample(images, min(SAMPLES_GRID, len(images)))

    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    loaded = 0
    for i, img_path in enumerate(sample):
        img = load_image(img_path)
        labels = load_labels(img_path)

        if img is None:
            axes[i].axis("off")
            continue

        draw_boxes(axes[i], img, labels, class_names)
        axes[i].set_title(f"{img_path.name[:20]}", fontsize=7)
        loaded += 1

    # Скрываем лишние ячейки
    for j in range(loaded, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Hard Hat Workers — примеры [{split}]\n"
        f"Зелёный = helmet (каска), Красный = head (нарушение)",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Сохранено: {save_path}")


def count_labels(images: list) -> dict:
    """Считает количество объектов каждого класса."""
    counts = {}
    for img_path in images:
        for cls_id, *_ in load_labels(img_path):
            counts[cls_id] = counts.get(cls_id, 0) + 1
    return counts


def plot_class_distribution(splits_counts: dict, class_names: list, save_path: Path):
    """Столбчатая диаграмма распределения классов по сплитам."""
    splits = list(splits_counts.keys())
    n_classes = len(class_names)

    x = np.arange(n_classes)
    width = 0.25
    colors_bar = ["#5599FF", "#FF8844", "#44CC88"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, split in enumerate(splits):
        counts = [splits_counts[split].get(j, 0) for j in range(n_classes)]
        bars = ax.bar(x + i * width, counts, width, label=split,
                      color=colors_bar[i % len(colors_bar)], alpha=0.85)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        str(cnt), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Класс", fontsize=12)
    ax.set_ylabel("Количество объектов", fontsize=12)
    ax.set_title("Распределение классов по выборкам", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(title="Выборка", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [OK] Сохранено: {save_path}")


def plot_bbox_size_distribution(images: list, save_path: Path):
    """Гистограмма размеров bounding boxes (нормализованные ширина/высота)."""
    widths, heights = [], []
    for img_path in images:
        for _, _, _, bw, bh in load_labels(img_path):
            widths.append(bw)
            heights.append(bh)

    if not widths:
        print("  [WARN] Нет меток для построения распределения размеров")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(widths, bins=40, color="#5599FF", alpha=0.8, edgecolor="white")
    axes[0].set_title("Ширина bounding box (нормализованная)", fontsize=11)
    axes[0].set_xlabel("Ширина (0–1)")
    axes[0].set_ylabel("Количество")
    axes[0].grid(alpha=0.3)

    axes[1].hist(heights, bins=40, color="#FF8844", alpha=0.8, edgecolor="white")
    axes[1].set_title("Высота bounding box (нормализованная)", fontsize=11)
    axes[1].set_xlabel("Высота (0–1)")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Распределение размеров боксов", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [OK] Сохранено: {save_path}")


def print_dataset_summary(data: dict, splits_info: dict):
    """Вывод сводки по датасету в консоль."""
    print("\n" + "=" * 55)
    print("  СВОДКА ПО ДАТАСЕТУ")
    print("=" * 55)
    print(f"  Классы: {data.get('names', [])}")
    print(f"  Количество классов: {data.get('nc', '?')}")
    print()
    for split, (imgs, counts) in splits_info.items():
        total_objs = sum(counts.values())
        print(f"  [{split.upper()}]")
        print(f"    Изображений : {len(imgs)}")
        print(f"    Объектов    : {total_objs}")
        for cls_id, cnt in sorted(counts.items()):
            cls_name = data["names"][cls_id] if cls_id < len(data["names"]) else cls_id
            pct = cnt / total_objs * 100 if total_objs else 0
            print(f"      {cls_name:<10}: {cnt:>5} ({pct:.1f}%)")
        print()
    print("=" * 55)


def main():
    print("=" * 55)
    print("  EDA: Детекция касок — анализ датасета")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Загружаем описание датасета
    print(f"\n[INFO] Загружаем {DATA_YAML}...")
    data = load_yaml(DATA_YAML)
    class_names = data.get("names", ["head", "helmet"])
    print(f"  Классы: {class_names}")

    # Собираем изображения по сплитам
    splits = ["train", "val", "test"]
    splits_info = {}
    splits_counts_all = {}

    for split in splits:
        img_dir = resolve_img_dir(data, split)
        if img_dir is None:
            print(f"  [SKIP] Сплит '{split}' не найден в data.yaml или на диске")
            continue

        images = collect_images(img_dir)
        if not images:
            print(f"  [SKIP] Изображения не найдены в {img_dir}")
            continue

        counts = count_labels(images)
        splits_info[split] = (images, counts)
        splits_counts_all[split] = counts
        print(f"  [OK] {split}: {len(images)} изображений")

    if not splits_info:
        print("\n[ОШИБКА] Не найдено ни одного сплита датасета.")
        print("         Проверьте dataset/data.yaml и структуру папок.")
        sys.exit(1)

    # Вывод сводки
    print_dataset_summary(data, splits_info)

    # Визуализации
    print("\n[PLOT] Строим графики...")

    # 1. Сетка примеров
    for split, (images, _) in splits_info.items():
        if images:
            plot_sample_grid(
                images, class_names, split,
                OUTPUT_DIR / f"samples_{split}.png"
            )

    # 2. Распределение классов
    if splits_counts_all:
        plot_class_distribution(
            splits_counts_all, class_names,
            OUTPUT_DIR / "class_distribution.png"
        )

    # 3. Распределение размеров боксов (по train)
    if "train" in splits_info:
        plot_bbox_size_distribution(
            splits_info["train"][0],
            OUTPUT_DIR / "bbox_size_distribution.png"
        )

    print(f"\n[DONE] Все графики сохранены в {OUTPUT_DIR}/")
    print("       Следующий шаг: python train.py")


if __name__ == "__main__":
    main()
