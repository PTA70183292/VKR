import streamlit as st
import requests
import pandas as pd
import uuid
import time

st.set_page_config(page_title="Анализ тональности", layout="wide")

API_URL = "http://localhost:8000"


if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Анализ"

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
        st.error(f"Ошибка при обращении к API: {str(e)}")
        return None

def upload_dataset(file):

    try:
        files = {"file": (file.name, file, "text/csv")}
        response = requests.post(
            f"{API_URL}/training/upload-dataset",
            files=files,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка загрузки: {str(e)}")
        return None

def start_training(dataset_path, num_epochs, batch_size, learning_rate):

    try:
        response = requests.post(
            f"{API_URL}/training/start",
            params={
                "dataset_path": dataset_path,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Ошибка запуска обучения: {str(e)}")
        return None

def get_training_status():

    try:
        response = requests.get(f"{API_URL}/training/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_training_history():

    try:
        response = requests.get(f"{API_URL}/training/history", timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return {"history": []}

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

# Заголовок
st.title("🎭 Система анализа тональности с обучением модели")

# Проверка API
api_status = check_api_health()

# Sidebar навигация
with st.sidebar:
    st.header("📋 Навигация")
    
    page = st.radio(
        "Выберите раздел:",
        ["Анализ текста", "Обучение модели", "История обучения", "История предсказаний"],
        key="page_selector"
    )
    
    st.markdown("---")
    st.markdown("**Статус API:**")
    if api_status:
        st.success("✅ Подключено")
    else:
        st.error("❌ Не подключено")
    
    st.markdown("---")
    st.text_input("ID пользователя", value=st.session_state.user_id, disabled=True)
    
    if st.button("🔄 Новый ID"):
        st.session_state.user_id = str(uuid.uuid4())
        st.rerun()

if not api_status:
    st.error("❌ API недоступен. Убедитесь, что FastAPI сервер запущен на http://localhost:8000")
    st.stop()


if page == "Анализ текста":
    st.header("🔍 Анализ тональности текста")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Введите текст для анализа:", 
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
            if st.button(f"📝 {label}", key=f"btn_{label}"):
                st.session_state.text_input = text
                st.rerun()
    
    if st.button("🔍 Анализировать", type="primary", use_container_width=True):
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
                        st.metric(label="Уровень уверенности", value=f"{score*100:.2f}%")
                    
                    with col2:
                        st.markdown("##### Информация о предсказании")
                        st.info(f"""
                        **ID записи:** {result['id']}  
                        **Метка:** {result['label']}  
                        **Время:** {result['created_at'][:19]}
                        """)
        else:
            st.warning("⚠️ Введите текст для анализа")


elif page == "Обучение модели":
    st.header("🎓 Обучение модели")
    
    # Проверяем статус обучения
    training_status = get_training_status()
    
    if training_status and training_status["is_training"]:
        st.warning("⚠️ Обучение уже выполняется")
        
        st.markdown("### Статус обучения")
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        status_placeholder.info(f"**Статус:** {training_status['status']}\n\n**Сообщение:** {training_status['message']}")
        
        if st.button("🔄 Обновить статус"):
            st.rerun()
    
    else:
        tab1, tab2 = st.tabs(["Загрузка датасета", "Параметры обучения"])
        
        with tab1:
            st.markdown("### 1. Загрузите датасет для обучения")
            st.info("📋 Датасет должен быть в формате CSV с колонками: `text` и `label`")
            
            # Пример формата
            with st.expander("📖 Показать пример формата датасета"):
                example_df = pd.DataFrame({
                    'text': [
                        'Отличный продукт!',
                        'Обычное качество',
                        'Ужасная покупка'
                    ],
                    'label': [0, 1, 2]
                })
                st.dataframe(example_df)
                st.caption("Метки: 0 - позитивный, 1 - нейтральный, 2 - негативный")
            
            uploaded_file = st.file_uploader(
                "Выберите CSV файл",
                type=['csv'],
                help="Файл должен содержать колонки 'text' и 'label'"
            )
            
            if uploaded_file is not None:
                if st.button("📤 Загрузить датасет", type="primary"):
                    with st.spinner("Загрузка датасета..."):
                        result = upload_dataset(uploaded_file)
                        
                        if result:
                            st.success("✅ Датасет успешно загружен!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Строк", result['rows'])
                            with col2:
                                st.metric("Колонок", len(result['columns']))
                            with col3:
                                st.write("**Распределение меток:**")
                                st.json(result['label_distribution'])
                            
                            st.session_state.dataset_path = result['path']
        
        with tab2:
            st.markdown("### 2. Настройте параметры обучения")
            
            if 'dataset_path' not in st.session_state:
                st.warning("⚠️ Сначала загрузите датасет во вкладке 'Загрузка датасета'")
            else:
                st.info(f"📁 Датасет: {st.session_state.dataset_path}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    num_epochs = st.slider("Количество эпох", 1, 10, 3)
                    batch_size = st.selectbox("Размер батча", [4, 8, 16, 32], index=1)
                
                with col2:
                    learning_rate = st.select_slider(
                        "Learning rate",
                        options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
                        value=2e-4,
                        format_func=lambda x: f"{x:.0e}"
                    )
                
                st.markdown("---")
                
                if st.button("🚀 Запустить обучение", type="primary", use_container_width=True):
                    result = start_training(
                        st.session_state.dataset_path,
                        num_epochs,
                        batch_size,
                        learning_rate
                    )
                    
                    if result:
                        st.success("✅ Обучение запущено!")
                        st.json(result)
                        time.sleep(2)
                        st.rerun()


elif page == "История обучения":
    st.header("История обучения моделей")
    
    history_data = get_training_history()
    
    if history_data["history"]:
        for idx, training in enumerate(history_data["history"]):
            with st.expander(f"🎓 Обучение #{idx + 1} - {training.get('timestamp', 'N/A')[:19]}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Эпох", training.get('num_epochs', 'N/A'))
                    st.metric("Размер батча", training.get('batch_size', 'N/A'))
                
                with col2:
                    st.metric("Learning Rate", f"{training.get('learning_rate', 0):.0e}")
                    st.metric("Train Loss", f"{training.get('train_loss', 0):.4f}")
                
                with col3:
                    st.metric("Train Samples", training.get('train_samples', 'N/A'))
                    st.metric("Eval Samples", training.get('eval_samples', 'N/A'))
                
                if 'model_path' in training:
                    st.info(f"📁 Путь к модели: `{training['model_path']}`")
    else:
        st.info("📭 История обучения пуста. Выполните обучение модели.")


elif page == "История предсказаний":
    st.header("История предсказаний")
    
    history = get_user_history(st.session_state.user_id, limit=50)
    
    if history:
        df_history = pd.DataFrame(history)
        

        df_history['sentiment_ru'] = df_history['label'].apply(
            lambda x: map_label_to_russian(x)[0]
        )
        df_history['emoji'] = df_history['label'].apply(
            lambda x: map_label_to_russian(x)[1]
        )
        
        # Статистика
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
        
        st.markdown("---")
        
        # Таблица
        display_df = df_history[['id', 'emoji', 'sentiment_ru', 'score', 'text', 'created_at']].copy()
        display_df['score'] = display_df['score'].apply(lambda x: f"{x*100:.2f}%")
        display_df['text'] = display_df['text'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
        display_df.columns = ['ID', '😊', 'Тональность', 'Уверенность', 'Текст', 'Время']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("История пуста. Выполните анализ текста.")

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Модель: BERT Multilingual + LoRA | Backend: FastAPI + PostgreSQL</small>
</div>
""", unsafe_allow_html=True)
