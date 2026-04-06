"""
train.py — Обучение нейронной сети для распознавания намерений.
Логирует: потери (loss), точность (accuracy), время каждой эпохи.
"""

import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import FeedForwardNet
from nltk_utils import bag_of_words, stem, tokenize

# Создаём папку для логов, если её нет
os.makedirs("logs", exist_ok=True)

# Настройка логирования: пишем одновременно в файл и в консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("TrainingLogger")


# Датасет для PyTorch — обёртка над numpy-массивами
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y).long()

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples


def prepare_data(intents_path: str = "intents.json"):
    """
    Читаем intents.json и превращаем паттерны в числовые векторы.
    Возвращает: обучающие данные X, метки y, словарь all_words, список тегов.
    """
    with open(intents_path, "r", encoding="utf-8") as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []  # пары (список_токенов, тег)

    for intent in intents["intents"]:
        tag = intent["tag"]
        if tag not in tags:
            tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    # Убираем знаки пунктуации, стеммируем и сортируем словарь
    ignore_chars = {"?", "!", ".", ",", ":", ";", "-", "'", '"', "(", ")"}
    all_words = sorted(set(stem(w) for w in all_words if w not in ignore_chars))
    tags = sorted(tags)

    # Превращаем паттерны в векторы bag-of-words
    X_train, y_train = [], []
    for (pattern_words, tag) in xy:
        bag = bag_of_words(pattern_words, all_words)
        X_train.append(bag)
        y_train.append(tags.index(tag))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)

    return X_train, y_train, all_words, tags


def train(
    intents_path: str = "intents.json",
    model_path: str = "trained_data.pth",
    hidden_size: int = 128,
    num_epochs: int = 500,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    dropout: float = 0.3,
):
    logger.info("=" * 60)
    logger.info("Начало обучения нейронной сети")
    logger.info(f"  Архитектура:     FeedForwardNet (2 скрытых слоя + BN + Dropout)")
    logger.info(f"  Скрытых нейронов: {hidden_size}")
    logger.info(f"  Эпох:            {num_epochs}")
    logger.info(f"  Размер батча:    {batch_size}")
    logger.info(f"  Learning rate:   {learning_rate}")
    logger.info(f"  Dropout:         {dropout}")
    logger.info("=" * 60)

    X_train, y_train, all_words, tags = prepare_data(intents_path)
    input_size = len(all_words)
    num_classes = len(tags)

    logger.info(f"Словарь: {input_size} слов | Намерений (классов): {num_classes}")
    logger.info(f"Обучающих примеров: {len(X_train)}")
    for tag in tags:
        logger.info(f"  Намерение: {tag}")

    dataset = ChatDataset(X_train, y_train)
    # shuffle=True — перемешиваем данные на каждой эпохе, чтобы сеть не запоминала порядок
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Устройство для обучения: {device}")

    model = FeedForwardNet(input_size, hidden_size, num_classes, dropout).to(device)

    # CrossEntropyLoss — стандартная функция потерь для задач классификации
    criterion = nn.CrossEntropyLoss()
    # Adam — адаптивный оптимизатор, хорошо работает "из коробки"
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # StepLR уменьшает learning rate в 2 раза каждые 200 эпох, чтобы сеть точнее сходилась
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    logger.info("\nНачало обучения:")
    logger.info(f"{'Эпоха':>7} | {'Loss':>10} | {'Accuracy':>10} | {'LR':>10} | {'Время':>8}")
    logger.info("-" * 60)

    start_total = time.time()
    avg_loss = 0.0
    accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()  # включаем режим обучения (BatchNorm и Dropout активны)
        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Прямой проход: получаем предсказания
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Обратный проход: обновляем веса
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        accuracy = correct / total * 100
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # Логируем первую эпоху, каждые 50 эпох и последнюю
        if epoch % 50 == 0 or epoch == 1 or epoch == num_epochs:
            logger.info(
                f"{epoch:>7} | {avg_loss:>10.4f} | {accuracy:>9.1f}% | {current_lr:>10.6f} | {elapsed:>6.2f}s"
            )

    total_time = time.time() - start_total
    logger.info("-" * 60)
    logger.info(f"Обучение завершено за {total_time:.1f} сек.")
    logger.info(f"Финальный loss: {avg_loss:.4f} | Точность: {accuracy:.1f}%")

    # Сохраняем модель и всё нужное для загрузки в chat.py
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": num_classes,
        "dropout": dropout,
        "all_words": all_words,
        "tags": tags,
    }
    torch.save(data, model_path)
    logger.info(f"Модель сохранена: {model_path}")
    logger.info("=" * 60)

    return model_path


if __name__ == "__main__":
    train()
