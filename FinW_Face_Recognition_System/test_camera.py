# Утилита для проверки доступности камеры или видеофайла.
# Полезно запускать перед основной системой, чтобы убедиться,
# что источник видео работает корректно.
# Использование: python test_camera.py [источник]
# Пример: python test_camera.py 0            (веб-камера)
#         python test_camera.py video.mp4     (видеофайл)

import cv2
import sys
import logging
from config import setup_logging, CAMERA_SOURCE

logger = setup_logging()


def test_camera(source=None):
    """
    Открывает источник видео, читает один кадр и выводит информацию о нём.
    Возвращает True если всё ок, False если что-то пошло не так.
    """
    if source is None:
        source = CAMERA_SOURCE

    logger.info("Проверяю источник видео: %s", source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Не могу открыть источник: %s", source)
        print(f"ОШИБКА: Источник '{source}' недоступен")
        return False

    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("Не могу прочитать кадр из: %s", source)
        print(f"ОШИБКА: Не удалось прочитать кадр из '{source}'")
        cap.release()
        return False

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"OK: Источник '{source}' доступен")
    print(f"  Разрешение: {w}x{h}")
    print(f"  FPS: {fps}")
    if total_frames > 0:
        # total_frames > 0 только для видеофайлов, для веб-камеры это -1
        print(f"  Всего кадров: {total_frames}")
        duration = total_frames / fps if fps > 0 else 0
        print(f"  Длительность: {duration:.1f} сек.")

    cap.release()
    logger.info("Проверка пройдена: %dx%d @ %.1f FPS", w, h, fps)
    return True


def main():
    # Если передан аргумент командной строки — используем его как источник
    source = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_camera(source)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
