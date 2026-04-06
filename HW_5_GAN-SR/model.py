import torch
import torch.nn as nn


# Простой residual блок. Суть: добавляем вход к выходу (skip connection).
# Это позволяет сети учиться "поверх" уже выученного — она не теряет информацию.
class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        # Складываем вход и выход — так градиент лучше проходит назад
        return x + self.block(x)


# Блок увеличения разрешения через Pixel Shuffle (sub-pixel convolution).
# Работает лучше, чем обычный Upsample+Conv, потому что учит апсемплинг напрямую.
# Идея: сначала увеличиваем количество каналов (x4), потом "перекладываем" их в пиксели.
class UpsampleBlock(nn.Module):
    def __init__(self, channels=64, scale=2):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),   # channels*4 -> channels, H*2 x W*2
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# Улучшенный генератор с residual блоками и Pixel Shuffle.
# Схема: Initial Conv -> ResBlocks -> Post-res Conv -> Upsample x2 -> Output Conv
class Generator(nn.Module):
    def __init__(self, num_res_blocks=6):
        super(Generator, self).__init__()

        # Входной слой — извлекаем начальные признаки
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Residual блоки — основная обработка
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(num_res_blocks)])

        # После residual блоков — ещё один conv с BatchNorm перед апсемплингом
        self.post_res = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
        )

        # Два блока апсемплинга: 16x16 -> 32x32 -> 64x64
        self.upsample1 = UpsampleBlock(64, scale=2)
        self.upsample2 = UpsampleBlock(64, scale=2)

        # Финальный слой: в RGB, нормировка в [-1, 1] через Tanh
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        init = self.initial(x)
        res = self.res_blocks(init)
        # Добавляем skip connection вокруг всего блока residual-цепочки
        x = init + self.post_res(res)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.output(x)
        return x


# Дискриминатор остаётся прежним — Conv + LeakyReLU, финальный Sigmoid.
# BatchNorm в дискриминаторе намеренно не используем на первом слое (стандартная практика).
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1 число
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x).view(-1)
