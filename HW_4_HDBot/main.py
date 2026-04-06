"""
main.py — Telegram-бот HelpDeskNeuro на aiogram 3.x.
При первом запуске автоматически обучает модель.
Параллельно поднимает Flask-сервер на порту 5000 для статус-страницы.
"""

import asyncio
import logging
import os
import sys
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template_string
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

import chat
from train import train

# Токен берём из переменной окружения — никогда не пишем его в код напрямую!
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not BOT_TOKEN:
    sys.exit("Ошибка: переменная окружения TELEGRAM_BOT_TOKEN не задана!")

MODEL_PATH = "trained_data.pth"
INTENTS_PATH = "intents.json"

# Общий лог бота: и в файл, и в консоль
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("BotLogger")


# ── Flask статус-страница ─────────────────────────────────────────────────────
# Простая веб-страница, чтобы убедиться что бот работает (можно открыть в браузере)
flask_app = Flask(__name__)

STATUS_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HelpDeskNeuro — Статус</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #0e0e14; color: #e2e8f0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.card { background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 16px; padding: 40px 48px; max-width: 480px; width: 100%; text-align: center; }
h1 { font-size: 28px; margin-bottom: 8px; color: #a78bfa; }
p { color: #94a3b8; margin-bottom: 24px; font-size: 15px; }
.badge { display: inline-block; background: #16a34a22; border: 1px solid #16a34a; color: #4ade80; padding: 6px 18px; border-radius: 999px; font-size: 13px; margin-bottom: 24px; }
.stats { background: #0e0e14; border-radius: 10px; padding: 16px; text-align: left; }
.stat { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1e1e30; font-size: 14px; }
.stat:last-child { border-bottom: none; }
.stat-label { color: #64748b; }
.stat-val { color: #c4b5fd; font-weight: 600; }
</style>
</head>
<body>
  <div class="card">
    <h1>HelpDeskNeuro</h1>
    <p>IT-ассистент на нейронной сети PyTorch для Telegram</p>
    <div class="badge">Работает</div>
    <div class="stats">
      <div class="stat"><span class="stat-label">Архитектура</span><span class="stat-val">FeedForward NN</span></div>
      <div class="stat"><span class="stat-label">Фреймворк</span><span class="stat-val">PyTorch</span></div>
      <div class="stat"><span class="stat-label">Намерений</span><span class="stat-val">{{ intents }}</span></div>
      <div class="stat"><span class="stat-label">Сообщений обработано</span><span class="stat-val">{{ messages }}</span></div>
      <div class="stat"><span class="stat-label">Модель обучена</span><span class="stat-val">{{ trained }}</span></div>
      <div class="stat"><span class="stat-label">Время запуска</span><span class="stat-val">{{ started }}</span></div>
    </div>
  </div>
</body>
</html>
"""

_start_time = datetime.now().strftime("%H:%M:%S %d.%m.%Y")


@flask_app.route("/")
def status_page():
    # Считаем количество обработанных сообщений по числу строк в chat.log
    msg_count = 0
    if os.path.exists("logs/chat.log"):
        with open("logs/chat.log", encoding="utf-8") as f:
            msg_count = sum(1 for _ in f)

    import json
    intent_count = 0
    if os.path.exists(INTENTS_PATH):
        with open(INTENTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
            intent_count = len(data.get("intents", []))

    return render_template_string(
        STATUS_HTML,
        intents=intent_count,
        messages=msg_count,
        trained="Да" if os.path.exists(MODEL_PATH) else "Обучается...",
        started=_start_time,
    )


@flask_app.route("/api/status")
def api_status():
    """JSON-эндпоинт для программной проверки статуса бота."""
    msg_count = 0
    if os.path.exists("logs/chat.log"):
        with open("logs/chat.log", encoding="utf-8") as f:
            msg_count = sum(1 for _ in f)
    return jsonify({
        "status": "running",
        "model_ready": os.path.exists(MODEL_PATH),
        "messages_handled": msg_count,
        "started": _start_time,
    })


def run_flask():
    """Запустить Flask в отдельном потоке (daemon — завершится вместе с ботом)."""
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# ── Telegram-обработчики ──────────────────────────────────────────────────────
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message):
    """Приветствие при первом запуске бота или команде /start."""
    name = message.from_user.first_name or "друг"
    logger.info(f"Новый пользователь: {message.from_user.id} (@{message.from_user.username})")
    await message.answer(
        f"Здравствуйте, {name}\\! Я *HelpDeskNeuro* — IT\\-ассистент службы поддержки\\.\n\n"
        "Опишите вашу IT\\-проблему, и я определю к какому специалисту обратиться "
        "и предложу варианты решения\\.\n\n"
        "Напишите /help чтобы узнать подробнее\\.",
        parse_mode="MarkdownV2",
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Справка по возможностям бота."""
    await message.answer(
        "*HelpDeskNeuro — справка*\n\n"
        "Я определяю нужного IT\\-специалиста по описанию проблемы:\n\n"
        "*Программист*\n"
        "1С, ошибки программ, базы данных, сайты, макросы, скрипты\n\n"
        "*Системный администратор*\n"
        "Сеть, интернет, пароли, принтеры, почта, VPN, вирусы, доступ к файлам\n\n"
        "*Электроник*\n"
        "Сломанное железо, монитор, питание, клавиатура, мышь, кабели, проектор\n\n"
        "Просто опишите проблему своими словами\\.",
        parse_mode="MarkdownV2",
    )


@dp.message(Command("status"))
async def cmd_status(message: Message):
    """Показать статус модели и счётчик сообщений."""
    model_exists = os.path.exists(MODEL_PATH)
    log_lines = 0
    if os.path.exists("logs/chat.log"):
        with open("logs/chat.log", encoding="utf-8") as f:
            log_lines = sum(1 for _ in f)
    await message.answer(
        f"*Статус HelpDeskNeuro*\n\n"
        f"Модель загружена: {'Да' if model_exists else 'Нет'}\n"
        f"Обработано сообщений: {log_lines}\n"
        f"Логи: `logs/training\\.log`, `logs/chat\\.log`, `logs/bot\\.log`",
        parse_mode="MarkdownV2",
    )


@dp.message(F.text)
async def handle_message(message: Message):
    """Основной обработчик: передаём текст в нейросеть и отправляем ответ."""
    user_text = message.text.strip()
    user_id = str(message.from_user.id)
    logger.info(f"[{user_id}] Получено: {user_text!r}")

    response = chat.get_response(user_text, user_id=user_id)

    # Пробуем отправить с Markdown-форматированием, при ошибке — простым текстом
    try:
        await message.answer(response, parse_mode="Markdown")
    except Exception:
        await message.answer(response)

    logger.info(f"[{user_id}] Отправлено: {response!r}")


# ── Точка входа ───────────────────────────────────────────────────────────────
async def main():
    # Если модель ещё не обучена — обучаем перед запуском
    if not os.path.exists(MODEL_PATH):
        logger.info("Файл модели не найден — запускаю обучение...")
        train(
            intents_path=INTENTS_PATH,
            model_path=MODEL_PATH,
            hidden_size=128,
            num_epochs=500,
            batch_size=16,
            learning_rate=0.001,
        )
    else:
        logger.info("Модель найдена, обучение пропускается.")

    logger.info("Загрузка модели в память...")
    chat.load_model(MODEL_PATH, INTENTS_PATH)
    logger.info("Модель загружена успешно.")

    # Flask запускаем в отдельном потоке, чтобы не блокировать бота
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Flask статус-сервер запущен на порту 5000.")

    logger.info("Запуск Telegram-бота HelpDeskNeuro...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
