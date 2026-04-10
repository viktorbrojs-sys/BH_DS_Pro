# Дашборд Streamlit для визуализации данных системы видеонаблюдения.
# Запуск: streamlit run streamlit_app.py
# Показывает статистику, графики активности и таблицу последних событий.

import streamlit as st
import pandas as pd
from datetime import datetime
from eda import load_events, compute_metrics

st.set_page_config(page_title="Дашборд видеонаблюдения", layout="wide")
st.title("Система видеонаблюдения — дашборд")

# Загружаем данные из базы
events = load_events(limit=1000)
metrics = compute_metrics(events)

# Четыре карточки с ключевыми числами наверху
col1, col2, col3, col4 = st.columns(4)
col1.metric("Всего событий", metrics["total_events"])
col2.metric("Опознано", metrics["known_events"])
col3.metric("Неизвестных", metrics["unknown_events"])
col4.metric("Процент распознавания", f"{metrics['recognition_rate']}%")

st.markdown("---")

if events:
    st.subheader("Активность по времени")
    hours_data = metrics["events_per_hour"]
    if hours_data:
        # Строим столбчатый график активности по часам
        df_hours = pd.DataFrame(
            list(hours_data.items()), columns=["Час", "Количество"]
        ).sort_values("Час")
        st.bar_chart(df_hours.set_index("Час"))

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Частота появления людей")
        freq = metrics["person_frequency"]
        if freq:
            # Топ людей по количеству появлений
            df_freq = pd.DataFrame(
                list(freq.items()), columns=["Человек", "Количество"]
            ).sort_values("Количество", ascending=False)
            st.bar_chart(df_freq.set_index("Человек"))

    with col_right:
        st.subheader("Известные vs Неизвестные")
        # Сравнение опознанных и неопознанных
        pie_data = pd.DataFrame({
            "Категория": ["Известные", "Неизвестные"],
            "Количество": [metrics["known_events"], metrics["unknown_events"]]
        })
        st.bar_chart(pie_data.set_index("Категория"))

    st.markdown("---")
    st.subheader("Последние события")

    # Таблица всех событий с читаемым временем
    df_events = pd.DataFrame(events)
    if "timestamp" in df_events.columns:
        df_events["время"] = df_events["timestamp"].apply(
            lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
        )
        df_events["статус"] = df_events["is_known"].apply(lambda x: "Известен" if x else "Неизвестен")
        display_cols = ["id", "person_name", "статус", "время", "zone_crossed"]
        available_cols = [c for c in display_cols if c in df_events.columns]
        st.dataframe(df_events[available_cols], use_container_width=True)
else:
    st.info("Событий пока нет. Запустите систему наблюдения, чтобы данные появились.")
