# Скрипт установки и первоначальной настройки системы видеонаблюдения.
# Создаёт нужные папки, проверяет зависимости, инициализирует базу данных
# и скачивает модели при первом запуске.
# Использование: python setup.py

import os
import sys
import subprocess
import urllib.request


def check_python_version():
    """Проверяем, что используется Python 3.9 или новее."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        print(f"Требуется Python 3.9+. Текущий: {major}.{minor}")
        sys.exit(1)
    print(f"Python {major}.{minor} — OK")


def create_directories():
    """Создаём рабочие папки, если они ещё не существуют."""
    dirs = [
        "known_faces",
        "captured_faces",
        "logs",
        "models",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Папка '{d}' — готова")


def create_env_file():
    """
    Создаём файл .env с настройками по умолчанию, если его нет.
    Пользователь должен заполнить токен бота и chat_id вручную.
    """
    env_path = ".env"
    if os.path.exists(env_path):
        print(f"Файл {env_path} уже существует — пропускаю")
        return

    content = """# Настройки системы видеонаблюдения
# Заполните TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID перед запуском

TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Источник видео: 0 — веб-камера, или путь к файлу (например: test_video.mp4)
CAMERA_SOURCE=0

# Полигон зоны наблюдения (список точек [[x,y], ...])
ZONE_POLYGON=[[100,200],[300,200],[300,400],[100,400]]

# Порог сравнения лиц (0.0 — строго, 1.0 — мягко)
FACE_MATCH_THRESHOLD=0.6

# Обрабатывать каждый N-й кадр (больше = меньше нагрузки)
PROCESS_EVERY_N_FRAMES=5

# True — использовать GPU (CUDA), False — CPU
USE_GPU=False

# Минимальная пауза между уведомлениями об одном человеке (секунды)
DEBOUNCE_SECONDS=5

# Тип трекера (на данный момент доступен только sort)
TRACKER_TYPE=sort

# Порт FastAPI сервера
API_PORT=8500
"""
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Файл {env_path} создан. Заполните токены Telegram перед запуском.")


def install_dependencies():
    """
    Устанавливает Python-зависимости из requirements.txt через pip.
    Запускает pip как подпроцесс, чтобы не зависеть от активного окружения.
    """
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"Файл {req_file} не найден — пропускаю установку зависимостей")
        return

    # Сначала ставим python-dotenv отдельно — он нужен самому setup.py
    subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"], capture_output=True)
    print("\nУстанавливаю зависимости...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file],
        capture_output=False
    )
    if result.returncode != 0:
        print("Ошибка при установке зависимостей. Проверьте вывод выше.")
        sys.exit(1)
    print("Зависимости установлены успешно.")


def download_face_model():
    """
    Скачивает SSD-модель для детекции лиц (OpenCV DNN).
    Файлы небольшие — около 10 МБ суммарно.
    Если уже скачаны — пропускает загрузку.
    """
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    proto_path = os.path.join(models_dir, "deploy.prototxt")
    model_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    if not os.path.exists(proto_path):
        print("Скачиваю конфигурацию детектора лиц...")
        urllib.request.urlretrieve(proto_url, proto_path)
        print("  deploy.prototxt — скачан")
    else:
        print("  deploy.prototxt — уже есть")

    if not os.path.exists(model_path):
        print("Скачиваю веса модели детектора лиц (~10 МБ)...")
        urllib.request.urlretrieve(model_url, model_path)
        print("  res10_300x300_ssd_iter_140000.caffemodel — скачан")
    else:
        print("  res10_300x300_ssd_iter_140000.caffemodel — уже есть")


def init_database():
    """
    Инициализирует базу данных SQLite.
    Создаёт таблицы persons и events, если их нет.
    """
    try:
        # Добавляем текущую папку в путь, чтобы импорты работали
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from database import init_db
        init_db()
        print("База данных SQLite инициализирована.")
    except Exception as e:
        print(f"Не удалось инициализировать базу данных: {e}")
        print("Попробуйте запустить: python database.py")


def print_next_steps():
    """Выводит инструкцию по следующим шагам после установки."""
    print("""
============================================================
Установка завершена! Следующие шаги:

1. Заполните .env — укажите TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID
   (или задайте их как переменные окружения)

2. Добавьте фотографии известных людей в папку known_faces/
   Имя файла = имя человека, например: Ivan_Petrov.jpg

3. Запустите обучение по фотографиям:
   python train_faces.py

4. Проверьте камеру:
   python test_camera.py

5. Запустите систему:
   python main.py

Дополнительно:
  python api_server.py          — REST API (порт 8500)
  streamlit run streamlit_app.py — веб-дашборд
  python eda.py                  — текстовый отчёт по метрикам
============================================================
""")


def main():
    print("=" * 60)
    print("Настройка системы видеонаблюдения")
    print("=" * 60)

    check_python_version()
    print()

    create_directories()
    print()

    create_env_file()
    print()

    install_dependencies()
    print()

    download_face_model()
    print()

    init_database()

    print_next_steps()


if __name__ == "__main__":
    main()
