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

st.title("Анализ тональности русских текстов")

try:
    tokenizer, model = load_model()
    st.success("Модель загружена")
except:
    st.error("Модель не найдена. Убедитесь, что модель обучена.")
    st.stop()

text_input = st.text_area("Введите текст для анализа:", height=150)

if st.button("Анализировать"):
    if text_input:
        with st.spinner("Анализ..."):
            sentiment, probs = predict_sentiment(text_input, tokenizer, model)
            
            st.subheader(f"Результат: {sentiment}")
            
            df = pd.DataFrame({
                'Класс': ['Позитивный', 'Нейтральный', 'Негативный'],
                'Вероятность': probs
            })
            
            st.bar_chart(df.set_index('Класс'))
            
            st.write("**Детальные вероятности:**")
            for idx, row in df.iterrows():
                st.write(f"{row['Класс']}: {row['Вероятность']:.2%}")
    else:
        st.warning("Пожалуйста, введите текст")
