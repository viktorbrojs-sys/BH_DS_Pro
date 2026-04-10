# Работа с базой данных SQLite.
# Здесь хранится информация об известных людях и все события пересечения зоны.

import sqlite3
import pickle
import time
import logging
import numpy as np
from config import DB_PATH

logger = logging.getLogger("surveillance.db")


def get_connection():
    """
    Открывает соединение с базой данных.
    check_same_thread=False нужен, чтобы соединение работало из разных потоков.
    row_factory позволяет обращаться к полям строки по имени, а не по индексу.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Создаёт таблицы при первом запуске. Если они уже существуют — ничего не делает.
    Таблица persons — база известных лиц.
    Таблица events — журнал всех пересечений зоны.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Таблица людей: имя, эмбеддинг лица в виде байт, контактная информация
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB,
            phone TEXT DEFAULT '',
            email TEXT DEFAULT '',
            additional_info TEXT DEFAULT '',
            created_at REAL DEFAULT (strftime('%s', 'now'))
        )
    """)

    # Таблица событий: кто, когда, в какой зоне и фото
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            person_name TEXT DEFAULT 'UNKNOWN',
            timestamp REAL DEFAULT (strftime('%s', 'now')),
            zone_crossed TEXT DEFAULT '',
            photo_path TEXT DEFAULT '',
            is_known INTEGER DEFAULT 0,
            FOREIGN KEY (person_id) REFERENCES persons(id)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("База данных инициализирована: %s", DB_PATH)


def add_person(name, embedding_vector, phone="", email="", additional_info=""):
    """
    Добавляет нового человека в базу.
    Эмбеддинг (вектор признаков лица) сериализуется через pickle и хранится как BLOB.
    Возвращает id новой записи.
    """
    conn = get_connection()
    cursor = conn.cursor()
    # Сериализуем вектор, чтобы положить его в BLOB-поле
    emb_blob = pickle.dumps(embedding_vector)
    cursor.execute(
        "INSERT INTO persons (name, embedding, phone, email, additional_info) VALUES (?, ?, ?, ?, ?)",
        (name, emb_blob, phone, email, additional_info)
    )
    conn.commit()
    person_id = cursor.lastrowid
    conn.close()
    logger.info("Добавлен человек '%s' с id=%d", name, person_id)
    return person_id


def get_all_persons():
    """
    Загружает всех людей из базы вместе с их эмбеддингами.
    Вызывается один раз при старте — эмбеддинги затем хранятся в памяти.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, embedding, phone, email, additional_info FROM persons")
    rows = cursor.fetchall()
    conn.close()

    persons = []
    for row in rows:
        # Десериализуем эмбеддинг обратно в список/массив
        emb = pickle.loads(row["embedding"]) if row["embedding"] else None
        persons.append({
            "id": row["id"],
            "name": row["name"],
            "embedding": emb,
            "phone": row["phone"],
            "email": row["email"],
            "additional_info": row["additional_info"]
        })
    return persons


def get_person_by_id(person_id):
    """
    Возвращает карточку одного человека по его id.
    Если не найден — возвращает None.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, phone, email, additional_info FROM persons WHERE id = ?", (person_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def add_event(person_id, person_name, zone_crossed, photo_path, is_known):
    """
    Записывает событие пересечения зоны в журнал.
    Для неизвестных людей person_id передаётся как None.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO events (person_id, person_name, timestamp, zone_crossed, photo_path, is_known) VALUES (?, ?, ?, ?, ?, ?)",
        (person_id, person_name, time.time(), zone_crossed, photo_path, int(is_known))
    )
    conn.commit()
    event_id = cursor.lastrowid
    conn.close()
    return event_id


def get_recent_events(limit=50):
    """
    Возвращает последние N событий, отсортированных от новых к старым.
    Используется в API и дашборде.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, person_id, person_name, timestamp, zone_crossed, photo_path, is_known FROM events ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_event_count():
    """Возвращает общее количество событий в базе."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM events")
    row = cursor.fetchone()
    conn.close()
    return row["cnt"]


def get_unknown_events(limit=50):
    """
    Возвращает события с неизвестными людьми для последующей разметки.
    Фото сохранены в папке captured_faces/.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, person_name, timestamp, photo_path FROM events WHERE is_known = 0 ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
