# GAN Super Resolution (PyTorch)

Реализация GAN для повышения разрешения изображений (16x16 -> 64x64), вдохновлённая архитектурой SRGAN.

---

## Датасет — Oxford 102 Flowers

Набор данных из 102 категорий цветов (Мария-Елена Нильсбак и Эндрю Зисерман).

Набор включает 102 вида цветов, которые часто встречаются в Великобритании. Каждый класс содержит от 40 до 258 изображений. Изображения сильно различаются по масштабу, позе и освещённости.

Датасет скачивается автоматически при первом запуске (~330 МБ).

---

## Структура проекта

```
gan_super_resolution/
    main.py        - точка входа, запуск обучения
    model.py       - Generator (ResBlocks + Pixel Shuffle) + Discriminator
    dataset.py     - загрузка датасета и подготовка данных
    train.py       - цикл обучения с метриками и LR scheduler
    utils.py       - визуализация, PSNR, SSIM, графики метрик
    inference.py   - применение обученной модели к своим фотографиям
    data/          - датасет (скачивается автоматически)
    results/       - сохранённые изображения, веса модели, metrics.csv
```

---

## Запуск

1. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

2. Запустить обучение:
   ```bash
   cd gan_super_resolution
   python3.10 main.py
   ```

   Или из корня проекта:
   ```bash
   python3.10 gan_super_resolution/main.py
   ```

---

## Что происходит при запуске

1. Автоматически скачивается датасет Oxford 102 Flowers (~330 МБ)
2. Берётся 300 изображений, каждое приводится к 64x64 (HR)
3. LR-версия создаётся уменьшением до 16x16
4. GAN обучается 60 эпох (~20–40 минут на CPU, быстрее на GPU)
5. Каждые 10 эпох сохраняется сравнительная сетка: **LR | Bilinear | GAN | HR**
6. После завершения строятся графики метрик и выводится итоговая таблица

---

## Архитектура

### Generator (~780K параметров)

```
Conv(3, 64) -> ReLU
    x6 ResBlock(64):  Conv -> BN -> ReLU -> Conv -> BN  (+skip)
Conv(64, 64) -> BN
Global skip connection (initial + post_res)
UpsampleBlock x2 (Pixel Shuffle, 16x16 -> 32x32 -> 64x64)
Conv(64, 3) -> Tanh
```

Каждый ResBlock добавляет вход к выходу — так градиенты лучше проходят назад и сеть сохраняет структуру изображения. Pixel Shuffle (sub-pixel convolution) учит апсемплинг напрямую — работает заметно лучше, чем обычный bilinear upsample.

### Discriminator (~2.7M параметров)

```
Conv(3, 64, stride=2) -> LeakyReLU               # 64x64 -> 32x32
Conv(64, 128, stride=2) -> BN -> LeakyReLU        # 32x32 -> 16x16
Conv(128, 256, stride=2) -> BN -> LeakyReLU       # 16x16 -> 8x8
Conv(256, 512, stride=2) -> BN -> LeakyReLU       # 8x8 -> 4x4
Conv(512, 1) -> Sigmoid
```

### Loss

```
G_loss = MSE(gen, hr)  +  5e-3 * BCE(D(gen), 1)
D_loss = (BCE(D(hr), 0.9) + BCE(D(gen), 0)) / 2
```

Content loss (MSE) следит за тем, чтобы сгенерированное изображение было близко к оригиналу. Adversarial loss (BCE) заставляет генератор обманывать дискриминатор — добавляет реалистичные детали.

---

## Обучение

- **Learning rate**: 1e-4 (G), 5e-5 (D)
- **LR Scheduler**: CosineAnnealingLR — плавно снижает LR к концу обучения, убирает осцилляции
- **Метрики**: PSNR и SSIM считаются каждую эпоху на фиксированном validation батче (гладкие кривые)
- **Baseline**: Bilinear upscale сравнивается с GAN по каждой метрике

---

## Метрики и результаты

После обучения в `results/` появятся:

| Файл | Содержимое |
|---|---|
| `epoch_010.png`, `epoch_020.png`, ... | Сравнительные сетки LR / Bilinear / GAN / HR |
| `metrics.csv` | PSNR, SSIM, G loss, D loss по каждой эпохе |
| `metrics_plot.png` | Графики метрик: loss, PSNR, SSIM |
| `generator.pth` | Веса обученного генератора |


---

## Применение к своим фотографиям

После обучения можно прогнать через генератор любые свои изображения:

```bash
# Одно фото
python3.10 inference.py --input my_photo.jpg

# Папка с фотографиями
python3.10 inference.py --input photos/

# Указать куда сохранить
python3.10 inference.py --input my_photo.jpg --output results/inference/
```

Результат — картинка с четырьмя колонками: **LR | Bilinear | GAN | Original**.

---

## Зависимости

```
torch >= 2.0.0
torchvision >= 0.15.0
Pillow >= 9.0.0
matplotlib >= 3.5.0
requests >= 2.28.0
numpy >= 1.23.0
scikit-image >= 0.19.0   # для расчёта SSIM
```
