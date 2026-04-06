"""
Используем обученный генератор на своих фотографиях.

Как запустить:
    python3.11 inference.py --input my_photo.jpg
    python3.11 inference.py --input photos/       # папка с изображениями
    python3.11 inference.py --input my_photo.jpg --output results/my_result.png

По умолчанию ищет веса в results/generator.pth
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import Generator


def load_generator(weights_path="results/generator.pth", device="cpu"):
    """
    Загружаем генератор из сохранённых весов.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Файл весов не найден: {weights_path}\n"
            "Сначала обучите модель: python3.11 main.py"
        )
    gen = Generator()
    gen.load_state_dict(torch.load(weights_path, map_location=device))
    gen.eval()
    gen.to(device)
    print(f"Веса загружены из {weights_path}")
    return gen


def process_image(generator, img_path, device="cpu"):
    """
    Обрабатываем одно изображение:
    - уменьшаем до 16x16 (имитируем LR)
    - прогоняем через генератор
    - возвращаем LR, GAN, Bilinear и оригинал для сравнения
    """
    img = Image.open(img_path).convert("RGB")

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # Оригинал масштабируем до 64x64 как HR-референс
    hr_pil = img.resize((64, 64), Image.BICUBIC)
    hr = normalize(to_tensor(hr_pil))

    # LR — 16x16
    lr_pil = img.resize((16, 16), Image.BICUBIC)
    lr = normalize(to_tensor(lr_pil))

    lr_batch = lr.unsqueeze(0).to(device)

    with torch.no_grad():
        gen_batch = generator(lr_batch)

    bilinear = F.interpolate(lr_batch, size=(64, 64), mode='bilinear', align_corners=False)

    return lr, gen_batch.squeeze(0).cpu(), bilinear.squeeze(0).cpu(), hr


def denormalize_np(tensor):
    """
    Тензор [-1,1] -> numpy (H, W, C) в [0, 1].
    """
    img = (tensor * 0.5 + 0.5).clamp(0, 1).cpu().detach().numpy()
    return np.transpose(img, (1, 2, 0))


def save_result(lr, gen, bilinear, hr, out_path):
    """
    Сохраняем сравнительную картинку: LR | Bilinear | GAN | Original.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["LR (16x16)", "Bilinear", "GAN (ours)", "Original HR"]
    images = [lr, bilinear, gen, hr]

    for ax, title, img_t in zip(axes, titles, images):
        ax.imshow(denormalize_np(img_t))
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Super Resolution Inference")
    parser.add_argument("--input", required=True, help="Путь к фото или папке с фотографиями")
    parser.add_argument("--output", default=None, help="Путь для сохранения результата")
    parser.add_argument(
        "--weights",
        default=os.path.join(os.path.dirname(__file__), "results", "generator.pth"),
        help="Путь к весам генератора",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    generator = load_generator(args.weights, device)

    input_path = Path(args.input)

    # Собираем список файлов
    if input_path.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [p for p in input_path.iterdir() if p.suffix.lower() in extensions]
        if not image_files:
            print(f"В папке {input_path} нет изображений")
            return
    elif input_path.is_file():
        image_files = [input_path]
    else:
        print(f"Файл или папка не найдены: {input_path}")
        return

    print(f"Обрабатываем {len(image_files)} изображений...")

    out_dir = Path(args.output) if args.output else Path(os.path.dirname(__file__)) / "results" / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        print(f"  -> {img_path.name}")
        try:
            lr, gen, bilinear, hr = process_image(generator, img_path, device)
            out_name = img_path.stem + "_sr.png"
            save_result(lr, gen, bilinear, hr, out_dir / out_name)
        except Exception as e:
            print(f"     Ошибка: {e}")

    print(f"\nВсе результаты сохранены в {out_dir}")


if __name__ == "__main__":
    main()
