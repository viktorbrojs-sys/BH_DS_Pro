# Конфигурация системы видеонаблюдения.
# Все параметры читаются из переменных окружения, чтобы не хранить
# чувствительные данные прямо в коде.
# Файл .env загружается автоматически через python-dotenv.

import os
import json
import logging

# Ищем .env в папке проекта и загружаем его до чтения os.getenv()
# Если python-dotenv не установлен — работаем без него
try:
    from dotenv import load_dotenv
    # Пробуем два места: папку файла config.py и текущую рабочую директорию
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(_env_path):
        _env_path = os.path.join(os.getcwd(), ".env")
    # override=True — значения из .env перекрывают переменные окружения системы
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass  # dotenv не установлен — продолжаем без него

# Токен бота и ID чата берутся из .env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Источник видео: 0 — веб-камера, или путь к видеофайлу (например, "video.mp4")
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
if CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)

# Полигон зоны наблюдения в формате JSON — список точек [[x,y], ...]
raw_polygon = os.getenv("ZONE_POLYGON", "[[100,200],[300,200],[300,400],[100,400]]")
ZONE_POLYGON = json.loads(raw_polygon)

# Порог совпадения лиц: чем меньше число, тем строже сравнение
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))

# Обрабатывать каждый N-й кадр, чтобы не перегружать процессор
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "5"))

# Использовать GPU (True) или CPU (False)
USE_GPU = os.getenv("USE_GPU", "False").lower() in ("true", "1", "yes")

# Минимальная пауза между повторными уведомлениями об одном человеке (в секундах)
DEBOUNCE_SECONDS = int(os.getenv("DEBOUNCE_SECONDS", "5"))

# Тип трекера — пока используется собственная реализация на основе центроидов
TRACKER_TYPE = os.getenv("TRACKER_TYPE", "sort")

# Пути к рабочим папкам проекта
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
CAPTURED_FACES_DIR = os.path.join(os.path.dirname(__file__), "captured_faces")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
DB_PATH = os.path.join(os.path.dirname(__file__), "surveillance.db")

# Максимальная ширина кадра перед обработкой — уменьшает нагрузку на CPU
FRAME_WIDTH = 640

# Хост и порт для FastAPI-сервера
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8500"))

# Создаём папки при первом запуске, если их нет
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(CAPTURED_FACES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "app.log")


def setup_logging():
    """
    Настраивает логирование: пишет одновременно в файл и в консоль.
    Возвращает корневой логгер системы.
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("surveillance")
