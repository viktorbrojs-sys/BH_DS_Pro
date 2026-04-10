# Определение зоны наблюдения и трекинг людей между кадрами.
# Зона задаётся полигоном (список точек). Трекер помогает не отправлять
# одно уведомление несколько раз подряд для одного и того же человека.

import numpy as np
import cv2
import logging

logger = logging.getLogger("surveillance.zone")


def point_in_polygon(point, polygon):
    """
    Проверяет, находится ли точка внутри заданного полигона.
    Использует встроенную функцию OpenCV — надёжно и быстро.

    point — кортеж (x, y), центр bounding box человека
    polygon — список точек вида [[x1,y1],[x2,y2],...]
    """
    polygon_np = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon_np, (int(point[0]), int(point[1])), False)
    return result >= 0


def draw_zone(frame, polygon, color=(0, 255, 255), thickness=2):
    """
    Рисует границы зоны наблюдения на кадре.
    По умолчанию — жёлтый цвет.
    """
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def get_bbox_center(x1, y1, x2, y2):
    """
    Вычисляет центральную точку bounding box.
    Используется для проверки, попал ли человек в зону.
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (cx, cy)


class SimpleTracker:
    """
    Простой трекер объектов на основе центроидов.

    Принцип работы: каждый обнаруженный человек получает уникальный id.
    При следующем кадре ищем ближайший объект из предыдущего кадра — это и
    есть "тот же" человек. Если объект не появляется max_disappeared кадров
    подряд — считаем, что он ушёл и удаляем его из трекера.
    """

    def __init__(self, max_disappeared=30):
        self.next_id = 0           # счётчик для выдачи новых id
        self.objects = {}          # словарь id -> координаты центра
        self.disappeared = {}      # счётчик пропавших кадров для каждого объекта
        self.max_disappeared = max_disappeared

    def update(self, detections):
        """
        Обновляет трекер по новым детекциям.
        detections — список bounding box'ов в формате (x1, y1, x2, y2).
        Возвращает словарь {id: (cx, cy)} всех активных объектов.
        """
        # Если в кадре никого нет — увеличиваем счётчик пропаж
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        # Вычисляем центры всех новых детекций
        input_centers = []
        for det in detections:
            cx = (det[0] + det[2]) / 2
            cy = (det[1] + det[3]) / 2
            input_centers.append((cx, cy))

        # Если раньше ничего не отслеживали — регистрируем всех как новых
        if len(self.objects) == 0:
            for center in input_centers:
                self.objects[self.next_id] = center
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centers = list(self.objects.values())

        # Строим матрицу расстояний: старые объекты x новые детекции
        dist_matrix = np.zeros((len(obj_centers), len(input_centers)))
        for i, oc in enumerate(obj_centers):
            for j, ic in enumerate(input_centers):
                dist_matrix[i][j] = np.sqrt((oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2)

        # Жадно сопоставляем объекты: ищем пары с минимальным расстоянием
        used_rows = set()
        used_cols = set()
        assignments = {}

        flat_indices = np.argsort(dist_matrix, axis=None)
        for idx in flat_indices:
            row = idx // len(input_centers)
            col = idx % len(input_centers)
            if row in used_rows or col in used_cols:
                continue
            # Если расстояние слишком большое — это уже другой человек
            if dist_matrix[row][col] > 150:
                continue
            assignments[row] = col
            used_rows.add(row)
            used_cols.add(col)

        # Обновляем позиции совпавших объектов или увеличиваем счётчик пропаж
        for row in range(len(obj_centers)):
            oid = obj_ids[row]
            if row in assignments:
                col = assignments[row]
                self.objects[oid] = input_centers[col]
                self.disappeared[oid] = 0
            else:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]

        # Детекции без пары — это новые люди в кадре
        for col in range(len(input_centers)):
            if col not in [assignments[r] for r in assignments]:
                self.objects[self.next_id] = input_centers[col]
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        return self.objects
