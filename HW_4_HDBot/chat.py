"""
chat.py — Загрузка обученной модели и получение ответа на сообщение пользователя.
Логирует каждый вопрос и ответ в logs/chat.log.
"""

import json
import logging
import os
from datetime import datetime

import torch
import torch.nn.functional as F

from model import FeedForwardNet
from nltk_utils import bag_of_words, tokenize

# Создаём папку для логов, если её нет
os.makedirs("logs", exist_ok=True)

# Отдельный логгер для вопросов/ответов — пишет только в chat.log
chat_logger = logging.getLogger("ChatLogger")
chat_logger.setLevel(logging.INFO)
if not chat_logger.handlers:
    fh = logging.FileHandler("logs/chat.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    chat_logger.addHandler(fh)

# Глобальные переменные — заполняются при вызове load_model()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_all_words = []
_tags = []
_intents = {}


def load_model(model_path: str = "trained_data.pth",
               intents_path: str = "intents.json"):
    """Загрузить обученную модель и список намерений в память."""
    global _model, _all_words, _tags, _intents

    data = torch.load(model_path, map_location=_device, weights_only=False)
    _all_words = data["all_words"]
    _tags = data["tags"]

    _model = FeedForwardNet(
        data["input_size"],
        data["hidden_size"],
        data["output_size"],
        data.get("dropout", 0.3),
    ).to(_device)
    _model.load_state_dict(data["model_state"])
    _model.eval()  # режим inference: отключаем Dropout и BatchNorm в режиме обучения

    with open(intents_path, "r", encoding="utf-8") as f:
        _intents = json.load(f)


def get_response(user_message: str,
                 user_id: str = "unknown",
                 threshold: float = 0.65) -> str:
    """
    Определить намерение пользователя и вернуть ответ.

    threshold — минимальная уверенность модели. Если уверенность ниже,
    возвращаем ответ для намерения 'unknown' (бот не понял вопрос).
    """
    import random

    if _model is None:
        return "Модель ещё не загружена. Попробуйте позже."

    # Превращаем сообщение в вектор bag-of-words
    tokens = tokenize(user_message)
    bow = bag_of_words(tokens, _all_words)

    X = torch.from_numpy(bow).unsqueeze(0).to(_device)

    with torch.no_grad():
        output = _model(X)
        # Softmax превращает логиты в вероятности (сумма = 1)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    conf_val = confidence.item()
    tag = _tags[predicted.item()]

    # Если уверенность ниже порога — считаем, что бот не понял вопрос
    if conf_val < threshold:
        tag = "unknown"

    # Ищем ответы для найденного тега и выбираем случайный
    response = "Не знаю как ответить на это."
    for intent in _intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            break

    # Подставляем время и дату, если шаблон их использует
    now = datetime.now()
    response = response.replace("{time}", now.strftime("%H:%M:%S"))
    response = response.replace("{date}", now.strftime("%d.%m.%Y (%A)"))

    # Логируем: кто спросил, какой тег нашли, уверенность, вопрос и ответ
    chat_logger.info(
        f"user={user_id} | tag={tag} | conf={conf_val:.3f} | "
        f"Q: {user_message!r} | A: {response!r}"
    )

    return response
