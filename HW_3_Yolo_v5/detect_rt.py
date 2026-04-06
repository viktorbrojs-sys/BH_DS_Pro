"""
Детекция касок в реальном времени с трекингом объектов.
Поддерживает: веб-камеру, видеофайл, изображение.
При обнаружении головы без каски выводится алерт "NO HELMET!".
Результат сохраняется как видео.

Запуск:
  python detect_rt.py                        — веб-камера (source=0)
  python detect_rt.py --source video.mp4     — видеофайл
  python detect_rt.py --source image.jpg     — изображение
  python detect_rt.py --weights best.pt      — свои веса
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path
from collections import deque

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Пути ────────────────────────────────────────────────────
YOLO_DIR    = Path("yolov5")
RUNS_DIR    = Path("runs")
OUTPUT_DIR  = Path("outputs")

# Автоматически ищем лучшие веса из последнего обучения
def find_best_weights() -> str:
    train_dir = RUNS_DIR / "train"
    if not train_dir.exists():
        return "yolov5m.pt"

    # Ищем самый свежий эксперимент
    experiments = sorted(train_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    for exp in reversed(experiments):
        best = exp / "weights" / "best.pt"
        if best.exists():
            return str(best)

    return "yolov5m.pt"
# ──────────────────────────────────────────────────────────────

# Подключаем YOLOv5
if YOLO_DIR.exists():
    sys.path.insert(0, str(YOLO_DIR))
else:
    print(f"[ОШИБКА] Папка {YOLO_DIR} не найдена. Запустите: python setup.py")
    sys.exit(1)

try:
    import torch
    import cv2
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
    from utils.augmentations import letterbox
    HAVE_YOLO = True
except ImportError as e:
    print(f"[ОШИБКА] Не удалось подключить YOLOv5: {e}")
    print("         Установите зависимости: python setup.py")
    sys.exit(1)


# ─── Цвета и параметры визуализации ──────────────────────────
COLOR_HELMET  = (50,  200,  50)   # BGR: зелёный  — каска есть
COLOR_HEAD    = (30,   30, 230)   # BGR: красный  — нарушение
COLOR_ALERT   = (0,    0,  220)   # BGR: ярко-красный для алерта
COLOR_INFO    = (255, 255, 255)   # BGR: белый
COLOR_BG      = (20,   20,  20)   # BGR: тёмный фон для текста
FONT          = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = 3                 # толщина рамки (было 2)
# ──────────────────────────────────────────────────────────────


def check_display() -> bool:
    """
    Проверяет, доступен ли графический дисплей.
    На сервере без X11/Wayland cv2.imshow упадёт — отключаем GUI заранее.
    """
    import os
    # На Linux без DISPLAY гарантированно headless
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    # Пробуем создать тестовое окно — если OpenCV собран без GUI, поймаем ошибку
    try:
        test = __import__("numpy").zeros((1, 1, 3), dtype="uint8")
        cv2.imshow("__test__", test)
        cv2.waitKey(1)
        cv2.destroyWindow("__test__")
        return True
    except cv2.error:
        return False


# Определяем режим один раз при старте
HEADLESS = not check_display()


class SimpleTracker:
    """
    Упрощённый трекер объектов без внешних зависимостей.
    Присваивает ID на основе IoU между кадрами (жадный матчинг).
    Для более точного трекинга рекомендуется SORT/DeepSORT.
    """

    def __init__(self, max_lost: int = 10, iou_threshold: float = 0.3):
        self.tracks = {}        # id -> {box, cls, lost, age}
        self.next_id = 1
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    @staticmethod
    def _iou(box_a, box_b) -> float:
        """Intersection over Union двух боксов [x1,y1,x2,y2]."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, detections: list) -> list:
        """
        detections: список (x1, y1, x2, y2, conf, cls_id)
        Возвращает список (x1, y1, x2, y2, conf, cls_id, track_id)
        """
        # Увеличиваем счётчик "потерянных" кадров для всех треков
        for tid in self.tracks:
            self.tracks[tid]["lost"] += 1

        matched_ids = set()
        results = []

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            det_box = (x1, y1, x2, y2)

            best_iou = self.iou_threshold
            best_tid = None

            for tid, track in self.tracks.items():
                if tid in matched_ids:
                    continue
                iou = self._iou(det_box, track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                # Обновляем существующий трек
                self.tracks[best_tid]["box"] = det_box
                self.tracks[best_tid]["cls"] = cls_id
                self.tracks[best_tid]["lost"] = 0
                self.tracks[best_tid]["age"] += 1
                matched_ids.add(best_tid)
                results.append((*det_box, conf, cls_id, best_tid))
            else:
                # Новый трек
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "box": det_box,
                    "cls": cls_id,
                    "lost": 0,
                    "age": 1
                }
                matched_ids.add(tid)
                results.append((*det_box, conf, cls_id, tid))

        # Удаляем потерянные треки
        lost_ids = [tid for tid, t in self.tracks.items()
                    if t["lost"] > self.max_lost]
        for tid in lost_ids:
            del self.tracks[tid]

        return results


class FPSCounter:
    """Скользящее среднее FPS за последние N кадров."""

    def __init__(self, window: int = 30):
        self.times = deque(maxlen=window)
        self._last = time.time()

    def tick(self) -> float:
        now = time.time()
        self.times.append(now - self._last)
        self._last = now
        if len(self.times) < 2:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))


def load_model(weights: str, device_str: str):
    """Загружает модель YOLOv5."""
    print(f"  [INFO] Загружаем модель: {weights}")
    device = select_device(device_str)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    model.warmup(imgsz=(1, 3, 640, 640))
    print(f"  [OK] Модель загружена на {device}")
    return model, device


def preprocess(frame, img_size: int = 640):
    """Подготовка кадра для инференса."""
    img = letterbox(frame, img_size, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC → CHW, BGR → RGB
    img = np.ascontiguousarray(img)
    return img


def draw_detection(frame, x1, y1, x2, y2, conf, cls_id, track_id, class_names):
    """Рисует один бокс детекции на кадре."""
    cls_name = class_names[cls_id] if cls_id < len(class_names) else "???"

    # Логика цвета: если в имени класса есть "helmet" — каска надета (зелёный),
    # иначе нарушение (красный). Так работает независимо от порядка классов в data.yaml.
    is_violation = "helmet" not in cls_name.lower()
    color = COLOR_HEAD if is_violation else COLOR_HELMET

    # Прямоугольник — толстая рамка для видимости
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # Подпись: ClassName #ID conf%
    label = f"{cls_name} #{track_id}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
    label_y = y1 - 6 if y1 > th + 10 else y2 + th + 8
    cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw + 6, label_y + 3), color, -1)
    cv2.putText(frame, label, (x1 + 3, label_y - 2),
                FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return is_violation


def draw_overlay(frame, fps: float, n_violations: int, total_frames: int):
    """Накладывает информационный HUD на кадр."""
    h, w = frame.shape[:2]

    # Верхняя полоска с FPS
    info_text = f"FPS: {fps:.1f}   Frame: {total_frames}"
    cv2.putText(frame, info_text, (10, 26), FONT, 0.65, COLOR_INFO, 1, cv2.LINE_AA)

    # Алерт при нарушениях
    if n_violations > 0:
        alert = f"! NO HELMET: {n_violations} {'нарушение' if n_violations == 1 else 'нарушений'} !"
        (aw, ah), _ = cv2.getTextSize(alert, FONT, 0.9, 2)
        ax = (w - aw) // 2
        ay = h - 30
        cv2.rectangle(frame, (ax - 8, ay - ah - 8), (ax + aw + 8, ay + 8),
                      COLOR_ALERT, -1)
        cv2.putText(frame, alert, (ax, ay), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def open_source(source):
    """
    Открывает источник видео.
    Возвращает (cap, is_webcam, is_image).
    """
    src_path = str(source)

    # Проверяем, это изображение?
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    if Path(src_path).suffix.lower() in img_exts:
        return None, False, True

    # Пробуем открыть как камеру (если source == "0" или целое число)
    if src_path.isdigit():
        cam_id = int(src_path)
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"[ОШИБКА] Не удалось открыть камеру #{cam_id}")
            print("         Проверьте, подключена ли камера.")
            sys.exit(1)
        return cap, True, False

    # Видеофайл
    if not Path(src_path).exists():
        print(f"[ОШИБКА] Файл не найден: {src_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"[ОШИБКА] Не удалось открыть файл: {src_path}")
        sys.exit(1)

    return cap, False, False


def process_image(model, device, img_path: str, class_names: list,
                  conf_thresh: float, iou_thresh: float, img_size: int,
                  output_dir: Path):
    """Инференс на одном изображении."""
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[ОШИБКА] Не удалось загрузить: {img_path}")
        return

    img = preprocess(frame, img_size)
    img_t = torch.from_numpy(img).to(device).float() / 255.0
    if img_t.ndim == 3:
        img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_t)
        pred = non_max_suppression(pred, conf_thresh, iou_thresh)

    tracker = SimpleTracker()

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_t.shape[2:], det[:, :4], frame.shape).round()
            dets_list = [(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls))
                         for x1, y1, x2, y2, conf, cls in det.tolist()]
            tracks = tracker.update(dets_list)
            for x1, y1, x2, y2, conf, cls_id, tid in tracks:
                draw_detection(frame, x1, y1, x2, y2, conf, cls_id, tid, class_names)

    out_path = output_dir / f"detected_{Path(img_path).name}"
    cv2.imwrite(str(out_path), frame)
    print(f"  [OK] Результат сохранён: {out_path}")

    if HEADLESS:
        print("  [INFO] Режим headless — окно не открывается, смотрите файл выше")
    else:
        cv2.imshow("Helmet Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(model, device, cap, is_webcam: bool, class_names: list,
                  conf_thresh: float, iou_thresh: float, img_size: int,
                  output_dir: Path, source: str, win_w: int = 960, win_h: int = 540,
                  max_frames: int = 0):
    """
    Обработка видеопотока (камера или файл) с трекингом и сохранением.

    Почему видео без буферизации играет 2x:
      Камера сообщает fps=30, но CPU обрабатывает ~5-10 fps.
      Пропущенные кадры не записываются, но VideoWriter помечает
      оставшиеся как 30fps → воспроизведение ускоряется.
    Решение: собираем кадры в буфер, в конце измеряем реальный fps
      и пишем видео с ним — скорость будет точной.
    """
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = "webcam_output.mp4" if is_webcam else f"detected_{Path(source).stem}.mp4"
    out_path = output_dir / out_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Для видеофайлов fps известен точно — буфер не нужен
    # Для камеры — буферизируем и пишем в конце с реальным fps
    use_buffer = is_webcam
    frame_buffer = []   # список готовых кадров (только для webcam)

    if not use_buffer:
        # Видеофайл: открываем VideoWriter сразу
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_source, (w, h))
    else:
        writer = None   # будет создан после цикла

    tracker = SimpleTracker()
    fps_counter = FPSCounter()
    frame_idx = 0
    total_violations = 0

    WIN_NAME = "Helmet Detection  |  q — выход"

    if not HEADLESS:
        # Создаём окно с возможностью изменения размера
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, win_w, win_h)

    if HEADLESS:
        print(f"\n[INFO] Режим headless — окно не открывается, прогресс каждые 50 кадров.")
        print(f"       Для остановки: Ctrl+C")
    else:
        print(f"\n[INFO] Запуск детекции... Нажмите 'q' для выхода.")
        print(f"       Окно можно свободно растягивать мышью.")
    if use_buffer:
        cap_hint = f" (буфер кадров, fps будет скорректирован по факту)"
    else:
        cap_hint = f" (fps источника: {fps_source:.1f})"
    print(f"       Результат{cap_hint}: {out_path}\n")

    t_start = time.time()

    try:
        while True:
            # Ограничение на число кадров (опционально)
            if max_frames > 0 and frame_idx >= max_frames:
                print(f"[INFO] Достигнут лимит кадров ({max_frames}).")
                break

            ret, frame = cap.read()
            if not ret:
                if not is_webcam:
                    print("[INFO] Видео завершено.")
                break

            frame_idx += 1
            fps = fps_counter.tick()

            # Инференс
            img = preprocess(frame, img_size)
            img_t = torch.from_numpy(img).to(device).float() / 255.0
            if img_t.ndim == 3:
                img_t = img_t.unsqueeze(0)

            with torch.no_grad():
                pred = model(img_t)
                pred = non_max_suppression(pred, conf_thresh, iou_thresh)

            # Детекции + трекинг
            n_violations = 0
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(img_t.shape[2:], det[:, :4], frame.shape).round()
                    dets_list = [(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls))
                                 for x1, y1, x2, y2, conf, cls in det.tolist()]
                    tracks = tracker.update(dets_list)
                    for x1, y1, x2, y2, conf, cls_id, tid in tracks:
                        is_viol = draw_detection(
                            frame, x1, y1, x2, y2, conf, cls_id, tid, class_names)
                        if is_viol:
                            n_violations += 1
                            total_violations += 1

            draw_overlay(frame, fps, n_violations, frame_idx)

            if use_buffer:
                frame_buffer.append(frame.copy())   # сохраняем для записи в конце
            else:
                writer.write(frame)

            if HEADLESS:
                if frame_idx % 50 == 0:
                    viol_mark = "  ⚠ НАРУШЕНИЕ" if n_violations > 0 else ""
                    print(f"  Кадр {frame_idx:>5} | FPS {fps:>5.1f}{viol_mark}", flush=True)
            else:
                cv2.imshow(WIN_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Остановлено пользователем.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Остановлено по Ctrl+C.")

    t_elapsed = time.time() - t_start
    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()

    # ── Запись видео с корректным FPS ──────────────────────────
    if use_buffer and frame_buffer:
        # Реальный fps = сколько кадров обработали за сколько секунд
        actual_fps = len(frame_buffer) / t_elapsed if t_elapsed > 0 else fps_source
        actual_fps = round(max(1.0, min(actual_fps, 60.0)), 2)
        print(f"\n[WRITE] Записываем {len(frame_buffer)} кадров с реальным FPS={actual_fps:.1f}...")
        writer = cv2.VideoWriter(str(out_path), fourcc, actual_fps, (w, h))
        for f in frame_buffer:
            writer.write(f)
        writer.release()
        print(f"  [OK] Скорость воспроизведения будет соответствовать реальной")
    elif writer is not None:
        writer.release()
    # ──────────────────────────────────────────────────────────

    print(f"\n[DONE] Обработано кадров : {frame_idx}")
    print(f"       Нарушений всего   : {total_violations}")
    print(f"       Видео сохранено   : {out_path}")


def parse_args():
    default_weights = find_best_weights()

    parser = argparse.ArgumentParser(
        description="Детекция касок в реальном времени (YOLOv5)"
    )
    parser.add_argument("--source",   type=str, default="0",
                        help="Источник: 0=камера, путь к видео/фото")
    parser.add_argument("--weights",  type=str, default=default_weights,
                        help=f"Путь к весам модели (по умолчанию: {default_weights})")
    parser.add_argument("--conf",     type=float, default=0.4,
                        help="Порог уверенности детекции [0..1]")
    parser.add_argument("--iou",      type=float, default=0.45,
                        help="Порог IoU для NMS [0..1]")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Размер входного изображения")
    parser.add_argument("--device",    type=str, default="",
                        help="Устройство: '0' (GPU), 'cpu' или '' (авто)")
    parser.add_argument("--classes",   type=str, default=None,
                        help="Имена классов через запятую (helmet,head)")
    parser.add_argument("--win-size",  type=str, default="960x540",
                        help="Начальный размер окна ШxВ (например 1280x720). "
                             "Окно можно растягивать мышью в любой момент.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Лимит кадров для веб-камеры (0 = без лимита). "
                             "Задайте, если память ограничена.")
    return parser.parse_args()


def main():
    print("=" * 55)
    print("  Детекция касок в реальном времени")
    print("=" * 55)

    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Определяем имена классов
    if args.classes:
        class_names = [c.strip() for c in args.classes.split(",")]
    else:
        # Пробуем загрузить из data.yaml
        yaml_path = Path("dataset/data.yaml")
        if yaml_path.exists():
            import yaml as _yaml
            with open(yaml_path) as f:
                d = _yaml.safe_load(f)
            class_names = d.get("names", ["head", "helmet"])
        else:
            class_names = ["head", "helmet"]

    display_mode = "headless (окно не откроется, только запись в файл)" if HEADLESS else "GUI (окно с видео)"

    print(f"\n[CONFIG]")
    print(f"  Источник    : {args.source}")
    print(f"  Веса        : {args.weights}")
    print(f"  Конфиденс   : {args.conf}")
    print(f"  Классы      : {class_names}")
    print(f"  Дисплей     : {display_mode}")

    print("\n[MODEL] Загрузка модели...")
    model, device = load_model(args.weights, args.device)

    # Открываем источник
    cap, is_webcam, is_image = open_source(args.source)

    # Парсим размер окна "960x540" → (960, 540)
    try:
        win_w, win_h = [int(x) for x in args.win_size.lower().split("x")]
    except Exception:
        print(f"[WARN] Неверный формат --win-size '{args.win_size}', используем 960x540")
        win_w, win_h = 960, 540

    if is_image:
        print("\n[MODE] Режим изображения")
        process_image(model, device, args.source, class_names,
                      args.conf, args.iou, args.img_size, OUTPUT_DIR)
    else:
        mode = "Веб-камера" if is_webcam else "Видеофайл"
        print(f"\n[MODE] {mode}")
        process_video(model, device, cap, is_webcam, class_names,
                      args.conf, args.iou, args.img_size, OUTPUT_DIR, args.source,
                      win_w=win_w, win_h=win_h, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
