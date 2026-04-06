import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from utils import save_sample_grid, compute_psnr, compute_ssim
import csv
import os


def train(generator, discriminator, dataset, num_epochs=60, batch_size=16,
          lr=1e-4, device="cpu", save_every=10):
    """
    Улучшенный цикл обучения GAN.

    Что изменилось по сравнению с предыдущей версией:
    - Фиксированный validation батч — метрики теперь не скачут
    - CosineAnnealingLR — learning rate плавно снижается, меньше осцилляций
    - Adversarial loss weight: 1e-3 -> 5e-3 (чуть больше давление на GAN)
    """

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Фиксированный validation батч — первые batch_size изображений, без шаффла.
    # Это ключевое изменение: теперь метрики считаются на одних и тех же картинках
    # каждую эпоху, поэтому кривые будут гладкими и сравнимыми.
    val_subset = Subset(dataset, list(range(min(batch_size, len(dataset)))))
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    lr_val_fixed, hr_val_fixed = next(iter(val_loader))
    lr_val_fixed = lr_val_fixed.to(device)
    hr_val_fixed = hr_val_fixed.to(device)
    bilinear_val_fixed = torch.nn.functional.interpolate(
        lr_val_fixed, size=(64, 64), mode='bilinear', align_corners=False
    )

    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.9, 0.999))

    # Cosine Annealing: LR плавно падает от lr до lr/100 за num_epochs итераций.
    # Это убирает осцилляции на поздних эпохах.
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=num_epochs, eta_min=lr / 100)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=num_epochs, eta_min=lr / 200)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    generator.to(device)
    discriminator.to(device)

    metrics = {
        "epoch": [],
        "g_loss": [],
        "d_loss": [],
        "psnr_gan": [],
        "psnr_bilinear": [],
        "ssim_gan": [],
        "ssim_bilinear": [],
    }

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "g_loss", "d_loss", "psnr_gan", "psnr_bilinear", "ssim_gan", "ssim_bilinear"])

    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()

        g_total = 0.0
        d_total = 0.0
        steps = 0

        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            bs = lr_imgs.size(0)

            real_labels = torch.full((bs,), 0.9, device=device)
            fake_labels = torch.full((bs,), 0.0, device=device)

            # --- Дискриминатор ---
            discriminator.zero_grad()

            pred_real = discriminator(hr_imgs)
            loss_real = bce_loss(pred_real, real_labels)

            with torch.no_grad():
                fake_imgs = generator(lr_imgs)
            pred_fake = discriminator(fake_imgs.detach())
            loss_fake = bce_loss(pred_fake, fake_labels)

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            opt_D.step()

            # --- Генератор ---
            generator.zero_grad()

            fake_imgs = generator(lr_imgs)
            pred_gen = discriminator(fake_imgs)

            content = mse_loss(fake_imgs, hr_imgs)
            adversarial = bce_loss(pred_gen, real_labels)

            # Немного увеличили вес adversarial loss — больше давления на реалистичность
            loss_G = content + 5e-3 * adversarial

            loss_G.backward()
            opt_G.step()

            g_total += loss_G.item()
            d_total += loss_D.item()
            steps += 1

        # Шагаем scheduler'ы
        scheduler_G.step()
        scheduler_D.step()

        avg_g = g_total / steps
        avg_d = d_total / steps

        # Считаем метрики на ФИКСИРОВАННОМ батче — кривые теперь гладкие
        generator.eval()
        with torch.no_grad():
            gen_val = generator(lr_val_fixed)

            psnr_g = compute_psnr(gen_val, hr_val_fixed)
            psnr_b = compute_psnr(bilinear_val_fixed, hr_val_fixed)
            ssim_g = compute_ssim(gen_val, hr_val_fixed)
            ssim_b = compute_ssim(bilinear_val_fixed, hr_val_fixed)

        print(
            f"Epoch [{epoch:3d}/{num_epochs}] "
            f"G: {avg_g:.4f}  D: {avg_d:.4f}  "
            f"PSNR GAN: {psnr_g:.2f} dB  Bilinear: {psnr_b:.2f} dB  "
            f"SSIM GAN: {ssim_g:.3f}  Bilinear: {ssim_b:.3f}  "
            f"lr: {scheduler_G.get_last_lr()[0]:.2e}"
        )

        metrics["epoch"].append(epoch)
        metrics["g_loss"].append(avg_g)
        metrics["d_loss"].append(avg_d)
        metrics["psnr_gan"].append(psnr_g)
        metrics["psnr_bilinear"].append(psnr_b)
        metrics["ssim_gan"].append(ssim_g)
        metrics["ssim_bilinear"].append(ssim_b)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_g:.6f}", f"{avg_d:.6f}",
                             f"{psnr_g:.4f}", f"{psnr_b:.4f}",
                             f"{ssim_g:.4f}", f"{ssim_b:.4f}"])

        if epoch % save_every == 0 or epoch == num_epochs:
            save_sample_grid(
                lr_val_fixed.cpu(),
                hr_val_fixed.cpu(),
                gen_val.cpu(),
                epoch=epoch,
            )

        generator.train()

    print("Обучение завершено.")
    print(f"Метрики сохранены в {csv_path}")
    return generator, metrics
