# Детекция и распознавание лиц.
# Используем OpenCV DNN (SSD на базе ResNet) — быстро работает на CPU без dlib.
# Для сравнения лиц применяем гистограммный дескриптор: цветовые и яркостные
# гистограммы дают достаточно надёжный результат для базового сравнения.

import os
import logging
import cv2
import numpy as np
from config import FACE_MATCH_THRESHOLD

logger = logging.getLogger("surveillance.face_matcher")

# Адреса для загрузки весов SSD-детектора лиц от OpenCV
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Локальные пути к файлам модели
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


def download_face_model():
    """
    Загружает файлы SSD-детектора лиц при первом запуске.
    Если файлы уже скачаны — пропускает загрузку.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(PROTO_PATH):
        logger.info("Скачиваю конфигурацию детектора лиц...")
        import urllib.request
        urllib.request.urlretrieve(PROTO_URL, PROTO_PATH)
    if not os.path.exists(CAFFEMODEL_PATH):
        logger.info("Скачиваю веса модели детектора лиц...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, CAFFEMODEL_PATH)


def compute_histogram_embedding(face_bgr, bins=64):
    """
    Вычисляет дескриптор лица на основе цветовых гистограмм.
    Не требует dlib и работает на CPU без GPU.

    Используем три цветовых пространства сразу:
    - HSV — хорошо отделяет цвет от яркости
    - Lab — более близко к восприятию человека
    - Grayscale — яркостная текстура

    Возвращает нормализованный вектор фиксированной длины.
    """
    # Приводим лицо к стандартному размеру для стабильного сравнения
    face_resized = cv2.resize(face_bgr, (128, 128))

    # Гистограммы в пространстве HSV
    hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    # Гистограммы в пространстве Lab
    lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2Lab)
    hist_l = cv2.calcHist([lab], [0], None, [bins], [0, 256])
    hist_a = cv2.calcHist([lab], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([lab], [2], None, [bins], [0, 256])

    # Гистограмма яркости (серый канал)
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    lbp_hist = cv2.calcHist([gray], [0], None, [bins * 2], [0, 256])

    # Склеиваем все гистограммы в один вектор и нормируем его
    embedding = np.concatenate([
        hist_h.flatten(), hist_s.flatten(), hist_v.flatten(),
        hist_l.flatten(), hist_a.flatten(), hist_b.flatten(),
        lbp_hist.flatten()
    ])
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


class FaceMatcher:
    """
    Класс для детекции и распознавания лиц.
    При создании автоматически загружает DNN-модель.
    Эмбеддинги известных лиц хранятся в памяти после load_known_faces().
    """

    def __init__(self):
        self.known_encodings = []  # вектора эмбеддингов известных лиц
        self.known_ids = []        # их id в базе данных
        self.known_names = []      # имена
        self.known_info = []       # полные карточки
        self.face_net = None
        self._init_face_detector()

    def _init_face_detector(self):
        """
        Загружает DNN-детектор лиц (SSD ResNet).
        Если загрузка не удалась — будет использован Haar-каскад как запасной вариант.
        """
        try:
            download_face_model()
            self.face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, CAFFEMODEL_PATH)
            logger.info("DNN-детектор лиц загружен успешно")
        except Exception as e:
            logger.error("Не удалось загрузить DNN-детектор: %s", str(e))
            self.face_net = None

    def load_known_faces(self, persons_list):
        """
        Загружает эмбеддинги из базы данных в оперативную память.
        Вызывается один раз при старте — не пересчитывается каждый кадр.
        """
        self.known_encodings = []
        self.known_ids = []
        self.known_names = []
        self.known_info = []

        for person in persons_list:
            if person["embedding"] is not None:
                self.known_encodings.append(np.array(person["embedding"]))
                self.known_ids.append(person["id"])
                self.known_names.append(person["name"])
                self.known_info.append(person)

        logger.info("Загружено %d известных лиц в память", len(self.known_encodings))

    def generate_embedding(self, face_image):
        """
        Вычисляет вектор признаков для переданного изображения лица.
        Принимает изображение в формате BGR (как читает cv2.imread).
        Возвращает None, если изображение пустое.
        """
        if face_image is None or face_image.size == 0:
            return None
        # Убеждаемся, что это цветное изображение (3 канала)
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            embedding = compute_histogram_embedding(face_image)
            return embedding
        return None

    def match_face(self, face_image_bgr):
        """
        Сравнивает лицо с базой известных людей.
        Использует косинусное расстояние между нормализованными векторами.

        Возвращает кортеж: (карточка_человека, расстояние, эмбеддинг).
        Если совпадения нет или база пуста — первый элемент будет None.
        """
        embedding = self.generate_embedding(face_image_bgr)
        if embedding is None:
            return None, None, None

        # Если база известных лиц пуста — сразу возвращаем "неизвестен"
        if len(self.known_encodings) == 0:
            return None, None, embedding

        best_dist = float("inf")
        best_idx = -1
        for i, known_emb in enumerate(self.known_encodings):
            # Косинусное расстояние: 0 — идентичные, 1 — полностью разные
            dot = np.dot(embedding, known_emb)
            dist = 1.0 - dot
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # Считаем совпадением только если расстояние меньше порога
        if best_dist < FACE_MATCH_THRESHOLD:
            person_info = self.known_info[best_idx]
            logger.info("Совпадение: '%s' (расстояние=%.3f)", person_info["name"], best_dist)
            return person_info, best_dist, embedding

        return None, best_dist, embedding

    def detect_faces_in_crop(self, person_crop_bgr):
        """
        Находит лица на вырезанном изображении человека.
        Возвращает список координат в формате (top, right, bottom, left).
        Если DNN недоступна — использует Haar-каскад.
        """
        if self.face_net is None:
            return self._detect_faces_cascade(person_crop_bgr)

        h, w = person_crop_bgr.shape[:2]

        # Подготавливаем входной блоб для нейросети
        blob = cv2.dnn.blobFromImage(
            cv2.resize(person_crop_bgr, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)  # среднее значение пикселей для нормализации
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                # Координаты лица приходят в долях от 0 до 1 — переводим в пиксели
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                # Возвращаем в формате (top, right, bottom, left)
                faces.append((y1, x2, y2, x1))

        return faces

    def _detect_faces_cascade(self, image_bgr):
        """
        Резервный детектор лиц на основе классического Haar-каскада.
        Менее точен, чем DNN, но не требует никаких дополнительных файлов.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = []
        for (x, y, w, h) in rects:
            faces.append((y, x + w, y + h, x))
        return faces
