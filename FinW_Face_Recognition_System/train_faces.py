# Пакетное добавление людей из папки с фотографиями.
#
# Поддерживает две структуры known_faces/:
#
# --- Режим 1: Подпапки (несколько фото на одного человека) ---
#
#   known_faces/
#       Ivan_Petrov/
#           front.jpg
#           side.jpg
#           glasses.jpg
#       Anna_Sidorova/
#           photo1.jpg
#           photo2.png
#
# Имя папки = имя человека (подчёркивания → пробелы).
# Все фото внутри папки добавляются как отдельные эмбеддинги — система
# при распознавании найдёт наиболее близкий из всех.
#
# --- Режим 2: Плоская структура (одно фото = один человек) ---
#
#   known_faces/
#       Ivan_Petrov.jpg
#       Anna_Sidorova.jpg
#
# Для обратной совместимости. Имя файла = имя человека.
#
# Оба режима можно смешивать в одной папке.
# Если человек уже есть в базе — папка/файл пропускаются целиком.

import os
import sys
import cv2
import logging
from config import setup_logging, KNOWN_FACES_DIR
from database import init_db, add_person, get_all_persons
from face_matcher import FaceMatcher

logger = setup_logging()
matcher = FaceMatcher()

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _process_photo(filepath, name):
    """
    Читает изображение, вычисляет эмбеддинг и добавляет в базу.
    Возвращает True при успехе, False при любой ошибке.
    """
    image = cv2.imread(filepath)
    if image is None:
        logger.warning("Не могу прочитать файл: %s", filepath)
        return False

    embedding = matcher.generate_embedding(image)
    if embedding is None:
        logger.warning("Лицо не найдено на фото: %s", filepath)
        return False

    add_person(name, embedding.tolist())
    return True


def train_from_directory(faces_dir=None):
    """
    Обходит known_faces/ и добавляет людей в базу.

    Подпапки (несколько фото): каждое фото даёт отдельный эмбеддинг.
    Файлы в корне (одно фото): один эмбеддинг на файл.

    Если человек уже есть в базе — пропускается, повторного добавления нет.
    Чтобы переобучить — удалите запись из базы и запустите заново.
    """
    if faces_dir is None:
        faces_dir = KNOWN_FACES_DIR

    if not os.path.exists(faces_dir):
        logger.error("Папка не найдена: %s", faces_dir)
        return

    existing = get_all_persons()
    existing_names = {p["name"] for p in existing}

    added_persons = 0
    added_photos = 0
    skipped = 0
    failed = 0

    entries = sorted(os.scandir(faces_dir), key=lambda e: e.name.lower())

    # === Режим 1: Подпапки (несколько фото на человека) ===
    for entry in entries:
        if not entry.is_dir():
            continue

        name = entry.name.replace("_", " ").strip()

        if name in existing_names:
            logger.info("Пропускаю '%s' — уже есть в базе", name)
            skipped += 1
            continue

        photo_files = sorted(
            f for f in os.listdir(entry.path)
            if f.lower().endswith(IMAGE_EXTS)
        )

        if not photo_files:
            logger.warning("Папка '%s' пуста или нет изображений — пропускаю", entry.name)
            continue

        logger.info("Обрабатываю '%s' (%d фото)...", name, len(photo_files))
        person_ok = False

        for filename in photo_files:
            filepath = os.path.join(entry.path, filename)
            ok = _process_photo(filepath, name)
            if ok:
                logger.info("  + %s", filename)
                added_photos += 1
                person_ok = True
            else:
                failed += 1

        if person_ok:
            added_persons += 1
            existing_names.add(name)
            logger.info("'%s' добавлен", name)

    # === Режим 2: Плоская структура (одно фото = один человек) ===
    for entry in entries:
        if entry.is_dir():
            continue
        if not entry.name.lower().endswith(IMAGE_EXTS):
            continue

        name = os.path.splitext(entry.name)[0].replace("_", " ").strip()

        if name in existing_names:
            logger.info("Пропускаю '%s' — уже есть в базе", name)
            skipped += 1
            continue

        ok = _process_photo(entry.path, name)
        if ok:
            logger.info("Добавлен '%s' из файла %s", name, entry.name)
            added_persons += 1
            added_photos += 1
            existing_names.add(name)
        else:
            failed += 1

    logger.info(
        "Обучение завершено: людей добавлено=%d, фото добавлено=%d, пропущено=%d, ошибок=%d",
        added_persons, added_photos, skipped, failed
    )
    print(f"\nРезультат:")
    print(f"  Людей добавлено:  {added_persons}")
    print(f"  Фото добавлено:   {added_photos}")
    print(f"  Пропущено:        {skipped}  (уже были в базе)")
    print(f"  Ошибок:           {failed}  (файл не читается или лицо не найдено)")


def main():
    init_db()
    faces_dir = sys.argv[1] if len(sys.argv) > 1 else KNOWN_FACES_DIR
    print(f"Обучение по фотографиям из: {faces_dir}")
    train_from_directory(faces_dir)


if __name__ == "__main__":
    main()
