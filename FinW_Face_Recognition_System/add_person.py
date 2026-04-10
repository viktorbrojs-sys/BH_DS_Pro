# Утилита для добавления нового человека в базу данных лиц.
# Запускается из командной строки, принимает фото и контактную информацию.
# Использование: python add_person.py <имя> <путь_к_фото> [телефон] [email] [инфо]

import sys
import os
import cv2
import logging
from config import setup_logging
from database import init_db, add_person
from face_matcher import FaceMatcher

logger = setup_logging()
matcher = FaceMatcher()


def add_person_from_photo(name, photo_path, phone="", email="", additional_info=""):
    """
    Читает фото, вычисляет эмбеддинг лица и записывает человека в базу.
    Если на фото нет лица — выводит ошибку и возвращает None.
    """
    if not os.path.exists(photo_path):
        logger.error("Фото не найдено: %s", photo_path)
        return None

    image = cv2.imread(photo_path)
    if image is None:
        logger.error("Не могу прочитать изображение: %s", photo_path)
        return None

    # Вычисляем дескриптор лица из фото
    embedding = matcher.generate_embedding(image)

    if embedding is None:
        logger.error("На фото не найдено лицо: %s", photo_path)
        return None

    person_id = add_person(name, embedding.tolist(), phone, email, additional_info)
    logger.info("Человек '%s' добавлен с id=%d", name, person_id)
    return person_id


def main():
    init_db()

    if len(sys.argv) < 3:
        print("Использование: python add_person.py <имя> <путь_к_фото> [телефон] [email] [инфо]")
        print("Пример: python add_person.py 'Иван Петров' known_faces/ivan.jpg '+375291234567' 'ivan@gmail.ru' 'Сотрудник'")
        sys.exit(1)

    name = sys.argv[1]
    photo_path = sys.argv[2]
    phone = sys.argv[3] if len(sys.argv) > 3 else ""
    email = sys.argv[4] if len(sys.argv) > 4 else ""
    info = sys.argv[5] if len(sys.argv) > 5 else ""

    person_id = add_person_from_photo(name, photo_path, phone, email, info)
    if person_id:
        print(f"Человек '{name}' успешно добавлен (id={person_id})")
    else:
        print("Не удалось добавить человека. Смотрите лог для подробностей.")


if __name__ == "__main__":
    main()
