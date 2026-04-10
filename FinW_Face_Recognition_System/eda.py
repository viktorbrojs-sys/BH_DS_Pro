# Анализ данных и метрики системы.
# Загружает события из базы данных и считает ключевые показатели:
# процент распознанных лиц, активность по часам, топ людей.
# Используется дашбордом Streamlit, но может запускаться и отдельно.

import sqlite3
import time
import logging
from datetime import datetime
from config import DB_PATH

logger = logging.getLogger("surveillance.eda")


def load_events(limit=500):
    """
    Загружает последние события из базы напрямую через SQLite.
    Не использует database.py намеренно — чтобы можно было вызывать
    изолированно, не инициализируя остальную систему.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, person_id, person_name, timestamp, zone_crossed, photo_path, is_known FROM events ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def compute_metrics(events):
    """
    Вычисляет метрики по списку событий.

    Возвращает словарь с:
    - total_events — всего событий
    - known_events — опознанных
    - unknown_events — неопознанных
    - recognition_rate — процент успешного распознавания
    - unique_persons — уникальных известных людей
    - events_per_hour — количество событий по часам (для графика)
    - person_frequency — сколько раз появился каждый человек
    """
    if not events:
        return {
            "total_events": 0,
            "known_events": 0,
            "unknown_events": 0,
            "recognition_rate": 0.0,
            "unique_persons": 0,
            "events_per_hour": {},
            "person_frequency": {}
        }

    total = len(events)
    known = sum(1 for e in events if e["is_known"])
    unknown = total - known
    rate = (known / total) * 100 if total > 0 else 0

    # Считаем уникальных опознанных людей
    unique_persons = len(set(e["person_name"] for e in events if e["is_known"]))

    events_per_hour = {}
    person_freq = {}
    for e in events:
        ts = e["timestamp"]
        # Группируем по часу для временного графика
        hour = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:00")
        events_per_hour[hour] = events_per_hour.get(hour, 0) + 1

        # Считаем частоту появления каждого человека
        name = e["person_name"]
        person_freq[name] = person_freq.get(name, 0) + 1

    return {
        "total_events": total,
        "known_events": known,
        "unknown_events": unknown,
        "recognition_rate": round(rate, 2),
        "unique_persons": unique_persons,
        "events_per_hour": events_per_hour,
        "person_frequency": person_freq
    }


def print_report():
    """
    Выводит текстовый отчёт по метрикам в консоль.
    Удобно для быстрой проверки без запуска дашборда.
    """
    events = load_events()
    metrics = compute_metrics(events)

    print("\n=== Отчёт по системе видеонаблюдения ===")
    print(f"Всего событий: {metrics['total_events']}")
    print(f"Опознанных: {metrics['known_events']}")
    print(f"Неизвестных: {metrics['unknown_events']}")
    print(f"Процент распознавания: {metrics['recognition_rate']}%")
    print(f"Уникальных известных людей: {metrics['unique_persons']}")

    if metrics["person_frequency"]:
        print("\nТоп по количеству появлений:")
        sorted_persons = sorted(metrics["person_frequency"].items(), key=lambda x: x[1], reverse=True)
        for name, count in sorted_persons[:10]:
            print(f"  {name}: {count}")

    if metrics["events_per_hour"]:
        print("\nАктивность по часам (последние записи):")
        sorted_hours = sorted(metrics["events_per_hour"].items())
        for hour, count in sorted_hours[-10:]:
            print(f"  {hour}: {count}")

    print("=" * 40)
    return metrics


if __name__ == "__main__":
    print_report()
