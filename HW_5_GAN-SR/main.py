import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from model import Generator, Discriminator
from dataset import download_flowers, FlowersDataset
from train import train
from utils import plot_metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")

    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    MAX_IMAGES = 100
    SAVE_EVERY = 10

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    images_dir = download_flowers(data_dir)

    dataset = FlowersDataset(images_dir=images_dir, max_images=MAX_IMAGES)

    generator = Generator()
    discriminator = Discriminator()

    print(f"Параметров генератора:     {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Параметров дискриминатора: {sum(p.numel() for p in discriminator.parameters()):,}")

    generator, metrics = train(
        generator=generator,
        discriminator=discriminator,
        dataset=dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        device=device,
        save_every=SAVE_EVERY,
    )

    # Сохраняем веса генератора
    weights_path = os.path.join(os.path.dirname(__file__), "results", "generator.pth")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(generator.state_dict(), weights_path)
    print(f"Веса генератора сохранены: {weights_path}")

    # Строим графики метрик
    plot_metrics(metrics, save_dir=os.path.join(os.path.dirname(__file__), "results"))

    print("\nГотово. Результаты в папке gan_super_resolution/results/")


if __name__ == "__main__":
    main()
