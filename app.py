import streamlit as st

st.set_page_config(page_title="Анализ тональности", page_icon="📊")

st.title("Анализ тональности текстов")

text_input = st.text_area("Введите текст:", height=150)

if st.button("Анализ"):
    if text_input:
        st.info("Загрузка модели...")
    else:
        st.warning("Bведите текст")
