import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd

st.set_page_config(page_title="Анализ тональности")

@st.cache_resource
def load_model():
    model_path = "./results/BERT_QLoRA"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    labels = {0: "Позитивный", 1: "Нейтральный", 2: "Негативный"}

    return labels[pred], probs[0].numpy()

st.title("Анализ тональности")

try:
    tokenizer, model = load_model()
    st.success("Модель успешно загружена")
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("Введите текст для анализа:", height=200, 
                              placeholder="Например: Отличный продукт, очень доволен покупкой!")

with col2:
    st.markdown("**Примеры для тестирования:**")
    examples = {
        "Позитивный": "Прекрасный отель, отличный сервис и замечательный персонал!",
        "Нейтральный": "Обычный отель, ничего особенного. Цена соответствует качеству.",
        "Негативный": "Ужасное обслуживание, грязные номера, не рекомендую никому."
    }
    
    for label, text in examples.items():
        if st.button(f"{label}", key=label):
            text_input = text
            st.rerun()

if st.button("Анализировать", type="primary"):
    if text_input:
        with st.spinner("Выполняется анализ..."):
            sentiment, probs, emoji = predict_sentiment(text_input, tokenizer, model)
            
            st.markdown("---")
            st.subheader(f"{emoji} Результат: **{sentiment}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Распределение вероятностей")
                df = pd.DataFrame({
                    'Класс': ['Позитивный', 'Нейтральный', 'Негативный'],
                    'Вероятность': probs
                })
                st.bar_chart(df.set_index('Класс'))
            
            with col2:
                st.markdown("##### Детальные значения")
                for idx, row in df.iterrows():
                    percentage = row['Вероятность'] * 100
                    st.metric(label=row['Класс'], value=f"{percentage:.2f}%")
    else:
        st.warning("Bведите текст для анализа")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Модель: BERT Multilingual (8-bit) + QLoRA | Датасет: ru_sentiment_dataset</small>
</div>
""", unsafe_allow_html=True)
