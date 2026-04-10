# Главный модуль системы видеонаблюдения.
# Запускает два параллельных потока:
#   1. Захват кадров с камеры/видео
#   2. Обработка кадров: YOLO -> детекция лиц -> сравнение с базой -> уведомление
# Между потоками используется очередь (queue) — первый поток не блокируется обработкой.

import sys
import os
import cv2
import time
import signal
import threading
import queue
import logging
import numpy as np
from datetime import datetime

from config import (
    setup_logging, CAMERA_SOURCE, ZONE_POLYGON, PROCESS_EVERY_N_FRAMES,
    USE_GPU, FRAME_WIDTH, CAPTURED_FACES_DIR, DEBOUNCE_SECONDS
)
from database import init_db, get_all_persons, add_event
from face_matcher import FaceMatcher
from zone_detector import SimpleTracker, point_in_polygon, draw_zone, get_bbox_center
from telegram_bot import send_notification

logger = setup_logging()

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Не установлен ultralytics. Выполните: pip install ultralytics")
    sys.exit(1)

# Очередь кадров между потоком захвата и потоком обработки
frame_queue = queue.Queue(maxsize=10)

# Очередь результатов — обработанные кадры для отображения
result_queue = queue.Queue(maxsize=10)

# Единственный экземпляр матчера и трекера — используются в обоих потоках
face_matcher = FaceMatcher()
tracker = SimpleTracker(max_disappeared=50)

# Блокировка для безопасного доступа к tracker.objects из разных потоков
tracker_lock = threading.Lock()

# Словарь: id трекера -> время последнего уведомления
# Защита от спама: не уведомляем чаще DEBOUNCE_SECONDS секунд на человека
# Два словаря: по tracker_id (точный) и по имени/метке (запасной, если трекер переназначил id)
last_notification_time = {}       # ключ: tracker_id
last_notification_by_label = {}   # ключ: "UNKNOWN" или имя человека

# Флаг работы системы — устанавливается в False для остановки всех потоков
_running = True


def _signal_handler(sig, frame):
    """
    Обработчик SIGINT (Ctrl+C). Устанавливает флаг остановки.
    Это надёжный способ выхода — работает даже когда окно cv2 не в фокусе.
    """
    global _running
    logger.info("Получен сигнал остановки (Ctrl+C). Завершаю работу...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)


def load_yolo_model():
    """
    Загружает модель YOLOv8s для детекции людей.
    Если USE_GPU=True — переносит модель на GPU (CUDA).
    """
    device = "cuda:0" if USE_GPU else "cpu"
    logger.info("Загружаю YOLO на устройство: %s", device)
    model = YOLO("yolov8s.pt")
    model.to(device)
    logger.info("YOLO загружена успешно")
    return model


def resize_frame(frame):
    """
    Уменьшает кадр до FRAME_WIDTH пикселей по ширине, сохраняя пропорции.
    Это снижает нагрузку на CPU при обработке без потери детекций.
    """
    h, w = frame.shape[:2]
    if w > FRAME_WIDTH:
        scale = FRAME_WIDTH / w
        new_w = FRAME_WIDTH
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    return frame


def video_capture_thread(source):
    """
    Поток захвата видео. Работает непрерывно и кладёт каждый N-й кадр
    в очередь frame_queue для обработки. None в конце означает конец потока.
    """
    global _running
    logger.info("Открываю источник видео: %s", source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Не могу открыть источник видео: %s", source)
        return

    frame_count = 0
    while cap.isOpened() and _running:
        ret, frame = cap.read()
        if not ret:
            logger.info("Видеопоток завершён или кадр недоступен")
            break

        frame = resize_frame(frame)
        frame_count += 1

        # Кладём в очередь только каждый N-й кадр — обрабатывать все нет смысла
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            if not frame_queue.full():
                frame_queue.put((frame.copy(), frame_count))

        time.sleep(0.01)

    cap.release()
    # Сигнализируем второму потоку, что кадры закончились
    frame_queue.put(None)
    logger.info("Поток захвата видео завершён")


def face_processing_thread(model):
    """
    Поток обработки кадров. Берёт кадры из очереди и выполняет:
    1. YOLO — находит людей в кадре
    2. Для каждого человека в зоне — ищет лицо через DNN-детектор
    3. Сравнивает лицо с базой известных людей
    4. Если нужно — сохраняет фото и отправляет уведомление в Telegram
    """
    logger.info("Запущен поток обработки лиц")
    global last_notification_time

    while True:
        item = frame_queue.get()
        # None — сигнал завершения работы
        if item is None:
            result_queue.put(None)
            break

        frame, frame_num = item
        detections_data = []

        try:
            # Запускаем YOLO, детектируем только класс 0 = "person"
            results = model(frame, verbose=False, classes=[0])
            boxes = results[0].boxes

            person_detections = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                # Отбрасываем слабые детекции
                if conf > 0.5:
                    person_detections.append((x1, y1, x2, y2, conf))

            # Обновляем трекер — это присваивает каждому человеку уникальный id
            with tracker_lock:
                tracker.update([(d[0], d[1], d[2], d[3]) for d in person_detections])

            for det in person_detections:
                x1, y1, x2, y2, conf = det
                center = get_bbox_center(x1, y1, x2, y2)
                in_zone = point_in_polygon(center, ZONE_POLYGON)

                # Находим tracker_id сразу — пока трекер не изменился
                with tracker_lock:
                    tracker_id = _find_tracker_id(center)

                # Базовая информация о детекции для последующей отрисовки
                # tracker_id сохраняем здесь — draw_results его только читает, гонки нет
                det_info = {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "in_zone": in_zone,
                    "person_info": None,
                    "is_known": False,
                    "face_bbox": None,
                    "tracker_id": tracker_id
                }

                # Детальную обработку делаем только если человек в зоне
                if in_zone:
                    # Вырезаем область человека из кадра
                    person_crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if person_crop.size == 0:
                        detections_data.append(det_info)
                        continue

                    # Ищем лицо внутри вырезанного изображения человека
                    face_locations = face_matcher.detect_faces_in_crop(person_crop)

                    if len(face_locations) > 0:
                        # Берём первое найденное лицо
                        top, right, bottom, left = face_locations[0]

                        # Переводим координаты лица обратно в систему координат кадра
                        face_x1 = max(0, x1 + left)
                        face_y1 = max(0, y1 + top)
                        face_x2 = min(frame.shape[1], x1 + right)
                        face_y2 = min(frame.shape[0], y1 + bottom)
                        det_info["face_bbox"] = (face_x1, face_y1, face_x2, face_y2)

                        # Вырезаем само лицо для сравнения с базой
                        face_crop = person_crop[top:bottom, left:right]
                        if face_crop.size == 0:
                            detections_data.append(det_info)
                            continue

                        matched_person, distance, encoding = face_matcher.match_face(face_crop)

                        # Определяем метку для этого человека (используется как запасной ключ дебаунса)
                        label = matched_person["name"] if matched_person else "UNKNOWN"
                        now = time.time()

                        # Проверка по tracker_id (точный трекинг)
                        blocked_by_tracker = (
                            tracker_id is not None and
                            now - last_notification_time.get(tracker_id, 0) < DEBOUNCE_SECONDS
                        )
                        # Запасная проверка по имени/метке (срабатывает когда трекер переназначил id)
                        blocked_by_label = (
                            now - last_notification_by_label.get(label, 0) < DEBOUNCE_SECONDS
                        )
                        should_notify = not blocked_by_tracker and not blocked_by_label

                        if matched_person:
                            # Человек опознан
                            det_info["person_info"] = matched_person
                            det_info["is_known"] = True

                            if should_notify:
                                photo_path = _save_face_image(frame, face_x1, face_y1, face_x2, face_y2, label)
                                add_event(matched_person["id"], label, "zone_1", photo_path, True)
                                send_notification(photo_path, matched_person, True)
                                if tracker_id is not None:
                                    last_notification_time[tracker_id] = now
                                last_notification_by_label[label] = now
                                logger.info("Известный человек: %s (расстояние=%.3f)", label, distance)
                        else:
                            # Человек неизвестен — сохраняем фото для разметки
                            if should_notify:
                                photo_path = _save_face_image(frame, face_x1, face_y1, face_x2, face_y2, "UNKNOWN")
                                add_event(None, "UNKNOWN", "zone_1", photo_path, False)
                                send_notification(photo_path, None, False)
                                if tracker_id is not None:
                                    last_notification_time[tracker_id] = now
                                last_notification_by_label["UNKNOWN"] = now
                                logger.info("Неизвестный человек обнаружен")

                detections_data.append(det_info)

        except Exception as e:
            logger.error("Ошибка при обработке кадра: %s", str(e))

        # Передаём результат в основной поток для отрисовки
        if not result_queue.full():
            result_queue.put((frame, detections_data))


def _find_tracker_id(center):
    """
    Находит id в трекере для объекта, ближайшего к заданной точке.
    Используется для антиспама: привязываем уведомления к конкретному человеку.
    Возвращает None, если никого рядом нет.
    ВАЖНО: вызывать только с удерживаемым tracker_lock.
    """
    best_id = None
    best_dist = float("inf")
    for tid, tc in tracker.objects.items():
        dist = np.sqrt((tc[0] - center[0]) ** 2 + (tc[1] - center[1]) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_id = tid
    if best_dist < 100:
        return best_id
    return None


def _save_face_image(frame, x1, y1, x2, y2, label):
    """
    Вырезает и сохраняет фото лица в папку captured_faces/.
    В имя файла включается имя (или UNKNOWN) и временная метка.
    """
    face_img = frame[max(0, y1):y2, max(0, x1):x2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_label = label.replace(" ", "_")
    filename = f"{safe_label}_{timestamp}.jpg"
    filepath = os.path.join(CAPTURED_FACES_DIR, filename)
    cv2.imwrite(filepath, face_img)
    return filepath


def draw_results(frame, detections_data):
    """
    Рисует на кадре всю отладочную информацию:
    - жёлтый полигон зоны наблюдения
    - зелёный bounding box если человек вне зоны, красный — если в зоне
    - синий прямоугольник вокруг найденного лица
    - подпись KNOWN/UNKNOWN над лицом
    - tracker_id в жёлтом цвете под bounding box (для отладки спама)
    """
    # Рисуем зону наблюдения
    draw_zone(frame, ZONE_POLYGON)

    for det in detections_data:
        x1, y1, x2, y2 = det["bbox"]
        in_zone = det["in_zone"]

        # Красный — человек в зоне, зелёный — вне зоны
        color = (0, 0, 255) if in_zone else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Отображаем tracker_id под bounding box — жёлтым цветом
        tid = det.get("tracker_id")
        id_text = f"ID:{tid}" if tid is not None else "ID:?"
        cv2.putText(frame, id_text, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if det["face_bbox"]:
            fx1, fy1, fx2, fy2 = det["face_bbox"]
            # Синий прямоугольник вокруг лица
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)

            if det["is_known"] and det["person_info"]:
                label = f"KNOWN: {det['person_info']['name']}"
                label_color = (0, 255, 0)
            else:
                label = "UNKNOWN"
                label_color = (0, 0, 255)
            cv2.putText(frame, label, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

    return frame


def main():
    """
    Точка входа: инициализирует базу данных, загружает известные лица,
    запускает два рабочих потока и основной цикл отрисовки.

    Выход из программы:
    - Нажать Q в окне видео (если окно в фокусе)
    - Ctrl+C в терминале (надёжный способ, работает всегда)
    """
    global _running
    logger.info("Инициализация системы видеонаблюдения")
    logger.info("Для выхода нажмите Ctrl+C в терминале или Q в окне видео")

    # Создаём базу данных при первом запуске
    init_db()

    # Загружаем известные лица из БД в память — один раз, при старте
    persons = get_all_persons()
    face_matcher.load_known_faces(persons)

    model = load_yolo_model()

    # Запускаем потоки как daemon — они завершатся вместе с основной программой
    cap_thread = threading.Thread(target=video_capture_thread, args=(CAMERA_SOURCE,), daemon=True)
    proc_thread = threading.Thread(target=face_processing_thread, args=(model,), daemon=True)

    cap_thread.start()
    proc_thread.start()

    logger.info("Система запущена. Обработка видео...")

    # Счётчик FPS для отображения на экране
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    # Проверяем, запущена ли система без дисплея (например, на сервере)
    _no_display = os.getenv("DISPLAY") is None and os.getenv("WAYLAND_DISPLAY") is None
    # Дополнительно проверяем, что OpenCV реально умеет показывать окна
    # (headless-сборка падает при cv2.imshow даже при наличии DISPLAY)
    if not _no_display:
        try:
            cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("__test__")
            headless = False
        except cv2.error:
            logger.warning("OpenCV собран без поддержки GUI — переключаюсь в headless-режим")
            headless = True
    else:
        headless = True

    while _running:
        try:
            item = result_queue.get(timeout=1.0)
        except queue.Empty:
            # Периодически проверяем флаг _running — нужно для своевременного выхода по Ctrl+C
            continue

        if item is None:
            break

        frame, detections_data = item
        frame = draw_results(frame, detections_data)

        # Обновляем счётчик FPS раз в секунду
        fps_counter += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_time = time.time()

        # Отображаем FPS в углу экрана
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Показываем окно только если есть дисплей
        if not headless:
            cv2.imshow("Surveillance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Нажата кнопка Q — завершаю работу")
                _running = False

    if not headless:
        cv2.destroyAllWindows()
    logger.info("Система остановлена")


if __name__ == "__main__":
    main()
