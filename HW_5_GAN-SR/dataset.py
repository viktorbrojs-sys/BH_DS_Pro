import os
import requests
import zipfile
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Скачиваем датасет Oxford 102 Flowers.
# Если файлы уже есть — пропускаем скачивание.
def download_flowers(data_dir="data"):
    data_dir = Path(data_dir)
    images_dir = data_dir / "jpg"

    if images_dir.exists() and len(list(images_dir.glob("*.jpg"))) > 0:
        print(f"Изображения уже есть в {images_dir}, скачивание пропущено.")
        return images_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    archive_path = data_dir / "102flowers.tgz"

    print("Скачиваем датасет Oxford 102 Flowers...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%", end="", flush=True)

    print("\nРаспаковываем архив...")

    # Это .tgz, распаковываем через tarfile
    import tarfile
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)

    archive_path.unlink()  # удаляем архив после распаковки
    print(f"Готово. Изображения в {images_dir}")
    return images_dir


# Датасет для задачи super resolution.
# HR — изображение 64x64, LR — уменьшенная копия 16x16.
class FlowersDataset(Dataset):
    def __init__(self, images_dir, max_images=100):
        self.image_paths = sorted(Path(images_dir).glob("*.jpg"))[:max_images]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"В папке {images_dir} не найдено ни одного .jpg файла")

        print(f"Загружено {len(self.image_paths)} изображений")

        # HR — оригинал 64x64, нормализован в [-1, 1]
        self.hr_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # LR — уменьшаем до 16x16
        self.lr_transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        hr = self.hr_transform(img)
        lr = self.lr_transform(img)
        return lr, hr
