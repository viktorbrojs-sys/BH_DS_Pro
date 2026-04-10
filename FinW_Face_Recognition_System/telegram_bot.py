# Отправка уведомлений в Telegram.
# Используем python-telegram-bot версии 20+ с асинхронным API.
# При пересечении зоны отправляется фото лица и карточка из базы данных.

import asyncio
import logging
import time
from telegram import Bot
from telegram.constants import ParseMode
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger("surveillance.telegram")

# Создаём бота один раз при запуске — повторно не создаём
bot = None
if TELEGRAM_BOT_TOKEN:
    bot = Bot(token=TELEGRAM_BOT_TOKEN)


async def send_notification_async(photo_path, person_info, is_known, zone_name="zone_1"):
    """
    Асинхронная функция отправки уведомления.
    Если человек опознан — отправляет полную карточку с контактами.
    Если неизвестен — сообщает об этом и предлагает просмотреть фото.

    photo_path — путь к сохранённому фото лица
    person_info — словарь с данными из базы (имя, телефон, email, доп. инфо)
    is_known — True если человек опознан
    zone_name — название зоны, которую пересекли
    """
    if not bot or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram не настроен, уведомление пропущено")
        return

    # Формируем текст сообщения в зависимости от того, известен ли человек
    if is_known and person_info:
        caption = (
            f"<b>Человек обнаружен в зоне</b>\n\n"
            f"<b>Имя:</b> {person_info.get('name', 'N/A')}\n"
            f"<b>Телефон:</b> {person_info.get('phone', 'N/A')}\n"
            f"<b>Email:</b> {person_info.get('email', 'N/A')}\n"
            f"<b>Инфо:</b> {person_info.get('additional_info', 'N/A')}\n"
            f"<b>Зона:</b> {zone_name}\n"
            f"<b>Время:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    else:
        caption = (
            f"<b>Неизвестный человек обнаружен</b>\n\n"
            f"<b>Зона:</b> {zone_name}\n"
            f"<b>Время:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Фото сохранено для последующей разметки."
        )

    try:
        with open(photo_path, "rb") as photo_file:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo_file,
                caption=caption,
                parse_mode=ParseMode.HTML
            )
        logger.info(
            "Уведомление Telegram отправлено: %s",
            person_info.get("name", "UNKNOWN") if person_info else "UNKNOWN"
        )
    except Exception as e:
        logger.error("Ошибка при отправке в Telegram: %s", str(e))


def send_notification(photo_path, person_info, is_known, zone_name="zone_1"):
    """
    Синхронная обёртка над асинхронной функцией отправки.
    Позволяет вызывать отправку из обычных (не async) потоков.
    Корректно работает как при наличии активного event loop, так и без него.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Если event loop уже запущен — планируем задачу в фоне
            asyncio.ensure_future(
                send_notification_async(photo_path, person_info, is_known, zone_name)
            )
        else:
            loop.run_until_complete(
                send_notification_async(photo_path, person_info, is_known, zone_name)
            )
    except RuntimeError:
        # Если loop недоступен — создаём новый
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            send_notification_async(photo_path, person_info, is_known, zone_name)
        )
