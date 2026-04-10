# REST API для доступа к данным системы видеонаблюдения.
# Построен на FastAPI — можно получить последние события, статистику
# и список неизвестных лиц для разметки.
# Запуск: python api_server.py (по умолчанию на порту 8500)

import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from database import init_db, get_recent_events, get_event_count, get_unknown_events
from config import API_HOST, API_PORT

logger = logging.getLogger("surveillance.api")

app = FastAPI(title="Surveillance API", version="1.0")


@app.get("/api/events")
def list_events(limit: int = 50):
    """
    Возвращает последние события пересечения зоны (известные и неизвестные).
    Параметр limit ограничивает количество записей.
    """
    events = get_recent_events(limit)
    return JSONResponse(content={"events": events, "total": get_event_count()})


@app.get("/api/events/unknown")
def list_unknown_events(limit: int = 50):
    """
    Возвращает только события с неизвестными людьми.
    Полезно для ручной разметки через интерфейс.
    """
    events = get_unknown_events(limit)
    return JSONResponse(content={"events": events})


@app.get("/api/stats")
def get_stats():
    """
    Сводная статистика: всего событий, из них известных и неизвестных
    за последние 100 записей.
    """
    total = get_event_count()
    recent = get_recent_events(100)
    known_count = sum(1 for e in recent if e["is_known"])
    unknown_count = sum(1 for e in recent if not e["is_known"])
    return JSONResponse(content={
        "total_events": total,
        "recent_known": known_count,
        "recent_unknown": unknown_count
    })


@app.get("/api/health")
def health():
    """Проверка работоспособности сервера."""
    return {"status": "ok"}


def run_api():
    """Инициализирует базу и запускает uvicorn-сервер."""
    import uvicorn
    init_db()
    logger.info("Запускаю API-сервер на %s:%d", API_HOST, API_PORT)
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")


if __name__ == "__main__":
    from config import setup_logging
    setup_logging()
    run_api()
