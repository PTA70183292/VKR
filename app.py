import streamlit as st
import requests
import pandas as pd
import uuid

st.set_page_config(page_title="Анализ тональности")

API_URL = "http://localhost:8000"

if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_sentiment(text: str, user_id: str):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"user_id": user_id, "text": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при обращении к API: {e}")
        return None

def get_user_history(user_id: str, limit: int = 10):
    try:
        response = requests.get(
            f"{API_URL}/predictions/user/{user_id}",
            params={"limit": limit},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except:
        return []

def map_label_to_russian(label: str) -> tuple:
    mapping = {
        "LABEL_0": ("Позитивный", "😊"),
        "LABEL_1": ("Нейтральный", "😐"),
        "LABEL_2": ("Негативный", "😞"),
        "positive": ("Позитивный", "😊"),
        "neutral": ("Нейтральный", "😐"),
        "negative": ("Негативный", "😞"),
    }
    return mapping.get(label, (label, "❓"))

st.title("Анализ тональности текста")

with st.spinner("Проверка подключения к API..."):
    api_status = check_api_health()

if api_status:
    st.success("API доступен")
else:
    st.error("API недоступен. Убедитесь, что FastAPI сервер запущен на http://localhost:8000")
    st.stop()

with st.sidebar:
    st.header("Настройки")
    st.text_input("User ID", value=st.session_state.user_id, disabled=True, 
                  help="Уникальный идентификатор пользователя")
    
    if st.button("Сгенерировать новый ID"):
        st.session_state.user_id = str(uuid.uuid4())
        st.rerun()
    
    st.markdown("---")
    st.markdown("**API Endpoint:**")
    st.code(API_URL, language="text")
    
    show_history = st.checkbox("Показать историю", value=True)

col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        height=200,
        placeholder="Например: Отличный продукт, очень доволен покупкой!",
        key="text_input"
    )

with col2:
    st.markdown("**Примеры для тестирования:**")
    examples = {
        "Позитивный": "Прекрасный отель, отличный сервис и замечательный персонал!",
        "Нейтральный": "Обычный отель, ничего особенного. Цена соответствует качеству.",
        "Негативный": "Ужасное обслуживание, грязные номера, не рекомендую никому."
    }
    
    for label, text in examples.items():
        if st.button(f"{label}", key=f"btn_{label}"):
            st.session_state.text_input = text
            st.rerun()

if st.button("Анализировать", type="primary", use_container_width=True):
    if text_input:
        with st.spinner("Выполняется анализ..."):
            result = predict_sentiment(text_input, st.session_state.user_id)
            
            if result:
                sentiment_ru, emoji = map_label_to_russian(result['label'])
                score = result['score']
                
                st.markdown("---")
                st.subheader(f"{emoji} Результат: **{sentiment_ru}**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Уверенность модели")
                    st.progress(score)
                    st.metric(label="Confidence Score", value=f"{score*100:.2f}%")
                
                with col2:
                    st.markdown("##### Информация о предсказании")
                    st.info(f"""
                    **ID записи:** {result['id']}  
                    **Метка:** {result['label']}  
                    **Время:** {result['created_at'][:19]}
                    """)
                
                with st.expander("📊 Подробная информация"):
                    st.json(result)
    else:
        st.warning("Введите текст для анализа")

if show_history:
    st.markdown("---")
    st.subheader("История предсказаний")
    
    history = get_user_history(st.session_state.user_id, limit=10)
    
    if history:
        df_history = pd.DataFrame(history)
        
        df_history['sentiment_ru'] = df_history['label'].apply(
            lambda x: map_label_to_russian(x)[0]
        )
        df_history['emoji'] = df_history['label'].apply(
            lambda x: map_label_to_russian(x)[1]
        )
        
        display_df = df_history[['id', 'emoji', 'sentiment_ru', 'score', 'text', 'created_at']].copy()
        display_df['score'] = display_df['score'].apply(lambda x: f"{x*100:.2f}%")
        display_df['text'] = display_df['text'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
        display_df.columns = ['ID', '😊', 'Тональность', 'Уверенность', 'Текст', 'Время']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            positive_count = len(df_history[df_history['sentiment_ru'] == 'Позитивный'])
            st.metric("😊 Позитивных", positive_count)
        
        with col2:
            neutral_count = len(df_history[df_history['sentiment_ru'] == 'Нейтральный'])
            st.metric("😐 Нейтральных", neutral_count)
        
        with col3:
            negative_count = len(df_history[df_history['sentiment_ru'] == 'Негативный'])
            st.metric("😞 Негативных", negative_count)
    else:
        st.info("История пуста. Выполните анализ текста, чтобы увидеть результаты здесь.")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Модель: BERT Multilingual (8-bit) + QLoRA | Backend: FastAPI + PostgreSQL</small>
</div>
""", unsafe_allow_html=True)
