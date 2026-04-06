import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os


def denormalize(tensor):
    """
    Из [-1, 1] обратно в [0, 1].
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def tensor_to_numpy(tensor):
    """
    Тензор (C, H, W) -> numpy (H, W, C) для matplotlib.
    """
    img = denormalize(tensor).cpu().detach().numpy()
    return np.transpose(img, (1, 2, 0))


def bilinear_upscale(lr_tensor):
    """
    Простой bilinear upscale LR -> 64x64. Baseline для сравнения с GAN.
    """
    return F.interpolate(
        lr_tensor.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False
    ).squeeze(0)


def compute_psnr(pred, target):
    """
    PSNR (Peak Signal-to-Noise Ratio) в дБ. Чем выше — тем лучше.
    Считаем в диапазоне [0, 1] после денормализации.
    Типичные значения: bilinear ~28-30 dB, хороший GAN > 30 dB.
    """
    pred_d = denormalize(pred)
    target_d = denormalize(target)
    mse = F.mse_loss(pred_d, target_d).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred, target):
    """
    Упрощённый SSIM (Structural Similarity Index). Диапазон [0, 1].
    Чем выше — тем лучше. Сравниваем структурное сходство изображений.
    Используем scikit-image для точного расчёта.
    """
    from skimage.metrics import structural_similarity as sk_ssim

    pred_np = denormalize(pred).cpu().detach().numpy()
    target_np = denormalize(target).cpu().detach().numpy()

    # Считаем SSIM по батчу и усредняем
    scores = []
    for i in range(pred_np.shape[0]):
        # (C, H, W) -> (H, W, C)
        p = np.transpose(pred_np[i], (1, 2, 0))
        t = np.transpose(target_np[i], (1, 2, 0))
        score = sk_ssim(p, t, data_range=1.0, channel_axis=2)
        scores.append(score)

    return float(np.mean(scores))


def save_sample_grid(lr_batch, hr_batch, generated_batch, epoch, save_dir="results", num_samples=4):
    """
    Сохраняем сравнительную сетку: LR | Bilinear | GAN | HR.
    """
    os.makedirs(save_dir, exist_ok=True)

    n = min(num_samples, lr_batch.size(0))

    fig = plt.figure(figsize=(4 * 4, n * 3))
    fig.suptitle(f"Epoch {epoch}: LR  |  Bilinear  |  GAN  |  HR", fontsize=14)

    for i in range(n):
        lr_img = tensor_to_numpy(lr_batch[i])
        hr_img = tensor_to_numpy(hr_batch[i])
        gen_img = tensor_to_numpy(generated_batch[i])
        bilinear_img = tensor_to_numpy(bilinear_upscale(lr_batch[i]))

        ax = fig.add_subplot(n, 4, i * 4 + 1)
        ax.imshow(lr_img)
        ax.set_title("LR (16x16)")
        ax.axis("off")

        ax = fig.add_subplot(n, 4, i * 4 + 2)
        ax.imshow(bilinear_img)
        ax.set_title("Bilinear")
        ax.axis("off")

        ax = fig.add_subplot(n, 4, i * 4 + 3)
        ax.imshow(gen_img)
        ax.set_title("GAN")
        ax.axis("off")

        ax = fig.add_subplot(n, 4, i * 4 + 4)
        ax.imshow(hr_img)
        ax.set_title("HR (64x64)")
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Сохранено: {save_path}")


def plot_metrics(metrics, save_dir="results"):
    """
    Строим три графика после обучения:
    1. Кривые лосса генератора и дискриминатора
    2. PSNR по эпохам: GAN vs Bilinear
    3. SSIM по эпохам: GAN vs Bilinear

    И выводим итоговую сравнительную таблицу в терминал.
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = metrics["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Metrics", fontsize=16)

    # --- График 1: Loss ---
    ax = axes[0]
    ax.plot(epochs, metrics["g_loss"], label="Generator Loss", color="blue")
    ax.plot(epochs, metrics["d_loss"], label="Discriminator Loss", color="red")
    ax.set_title("Loss по эпохам")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- График 2: PSNR ---
    ax = axes[1]
    ax.plot(epochs, metrics["psnr_gan"], label="GAN", color="green", linewidth=2)
    ax.plot(epochs, metrics["psnr_bilinear"], label="Bilinear", color="orange",
            linewidth=2, linestyle="--")
    ax.set_title("PSNR по эпохам (dB, выше = лучше)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- График 3: SSIM ---
    ax = axes[2]
    ax.plot(epochs, metrics["ssim_gan"], label="GAN", color="green", linewidth=2)
    ax.plot(epochs, metrics["ssim_bilinear"], label="Bilinear", color="orange",
            linewidth=2, linestyle="--")
    ax.set_title("SSIM по эпохам (0–1, выше = лучше)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "metrics_plot.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"\nГрафики метрик сохранены: {plot_path}")

    # --- Итоговая таблица в терминале ---
    last = -1
    print("\n" + "=" * 60)
    print("  Итоговое сравнение (последняя эпоха)")
    print("=" * 60)
    print(f"  {'Метрика':<20} {'GAN':>12} {'Bilinear':>12}")
    print(f"  {'-'*44}")
    print(f"  {'PSNR (dB)':<20} {metrics['psnr_gan'][last]:>12.2f} {metrics['psnr_bilinear'][last]:>12.2f}")
    print(f"  {'SSIM':<20} {metrics['ssim_gan'][last]:>12.3f} {metrics['ssim_bilinear'][last]:>12.3f}")
    print(f"  {'G Loss':<20} {metrics['g_loss'][last]:>12.4f} {'—':>12}")
    print(f"  {'D Loss':<20} {metrics['d_loss'][last]:>12.4f} {'—':>12}")
    print("=" * 60)

    # Сравниваем с bilinear
    psnr_diff = metrics["psnr_gan"][last] - metrics["psnr_bilinear"][last]
    ssim_diff = metrics["ssim_gan"][last] - metrics["ssim_bilinear"][last]
    print(f"\n  GAN vs Bilinear:")
    sign_p = "+" if psnr_diff >= 0 else ""
    sign_s = "+" if ssim_diff >= 0 else ""
    print(f"    PSNR: {sign_p}{psnr_diff:.2f} dB")
    print(f"    SSIM: {sign_s}{ssim_diff:.3f}")
    print()
