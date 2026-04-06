import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    """
    Полносвязная нейронная сеть с BatchNorm и Dropout.

    Архитектура:
        Input -> FC1 -> BN -> ReLU -> Dropout
               -> FC2 -> BN -> ReLU -> Dropout
               -> FC3 (выход — логиты по числу намерений)

    Почему Feed-Forward Network?
        Задача — классификация намерений (intent classification).
        Вход — вектор bag-of-words фиксированной длины, поэтому рекуррентные
        сети (LSTM, GRU) избыточны. FNN справляется быстро и хорошо.

        BatchNorm ускоряет обучение и стабилизирует градиенты.
        Dropout уменьшает переобучение (сеть "не заучивает" обучающие примеры).
        Выходной слой возвращает логиты — без softmax, потому что
        CrossEntropyLoss сам применяет softmax внутри.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 dropout: float = 0.3):
        super(FeedForwardNet, self).__init__()
        self.net = nn.Sequential(
            # Скрытый слой 1
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # Скрытый слой 2
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # Выходной слой: один нейрон на каждое намерение
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
