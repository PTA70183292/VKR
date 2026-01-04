import streamlit as st
import requests
import pandas as pd
import uuid
import time
import os

if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = None

if "dataset_uploaded" not in st.session_state:
    st.session_state.dataset_uploaded = False

st.set_page_config(
    page_title="Анализ тональности (BERT)", 
    layout="wide",
    page_icon="🎭"
)

API_URL = os.getenv("BACKEND_URL", "http://backend:8000")

def set_text(text_to_set):
    st.session_state.text_input = text_to_set

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_sentiment(text: str, user_id: str, model_name: str = None):
    try:
        payload = {
            "user_id": user_id, 
            "text": text,
            "model_name": model_name  # <-- Передаем выбранную модель
        }
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка API: {str(e)}")
        return None

def upload_dataset(file):
    try:
        files = {"file": (file.name, file, "text/csv")}
        response = requests.post(
            f"{API_URL}/training/upload-dataset",
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()

        try:
            detail = response.json().get("detail", response.text)
        except:
            detail = response.text

        st.error(f"Ошибка загрузки датасета:\n{detail}")
        return None

    except Exception as e:
        st.error(f"Ошибка соединения с backend: {str(e)}")
        return None

def get_available_models():
    try:
        response = requests.get(f"{API_URL}/training/models-list", timeout=5)
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except:
        return []

def start_training(dataset_path, num_epochs, batch_size, learning_rate, custom_model_name, source_model_path):
    if not dataset_path:
        st.error("dataset_path отсутствует, обучение не запущено")
        return None

    payload = {
        "dataset_path": dataset_path,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "custom_model_name": custom_model_name,
        "source_model_path": source_model_path
    }

    try:
        response = requests.post(
            f"{API_URL}/training/start",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            return response.json()

        detail = response.json().get("detail", response.text)
        st.error(f"Ошибка запуска обучения: {detail}")
        return None

    except Exception as e:
        st.error(f"Ошибка соединения с API: {str(e)}")
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
        "LABEL_0": ("Нейтральный", "😐"),
        "LABEL_1": ("Позитивный", "😊"),
        "LABEL_2": ("Негативный", "😞"),
        "neutral": ("Нейтральный", "😐"),
        "positive": ("Позитивный", "😊"),
        "negative": ("Негативный", "😞"),
    }
    return mapping.get(label, (label, "❓"))

query_params = st.query_params

if "user_id" in query_params:
    st.session_state.user_id = query_params["user_id"]
else:
    new_user_id = str(uuid.uuid4())
    st.session_state.user_id = new_user_id
    st.query_params["user_id"] = new_user_id

st.title("🎭 Система анализа тональности (Deep Learning)")

api_status = check_api_health()

with st.sidebar:
    st.header("📋 Меню")
    page = st.radio(
        "Разделы:",
        ["Анализ текста", "Обучение модели", "История обучения", "История предсказаний"],
        key="page_selector"
    )
    
    st.markdown("---")
    st.markdown("**Состояние системы:**")
    if api_status:
        st.success(f"✅ API доступен\n\n`{API_URL}`")
    else:
        st.error(f"❌ API недоступен\n\n`{API_URL}`")
    
    st.markdown("---")
    st.text_input("Ваш User ID", value=st.session_state.user_id, disabled=True)
    if st.button("🔄 Сбросить ID"):
        new_user_id = str(uuid.uuid4())
        st.session_state.user_id = new_user_id
        st.query_params["user_id"] = new_user_id
        st.rerun()

if not api_status:
    st.warning("⚠️ Не удается подключиться к Бэкенду. Проверьте логи Docker контейнера.")
    st.stop()

if page == "Анализ текста":
    st.header("🔍 Анализ тональности")
    
    # ПОЛУЧАЕМ СПИСОК МОДЕЛЕЙ
    available_models = get_available_models()
    
    # ВЫБОР МОДЕЛИ В ИНТЕРФЕЙСЕ
    # Добавляем "Базовая" в начало списка
    model_options = ["Default (Предобученная LoRA)"] + available_models
    
    col_sel, col_space = st.columns([1, 2])
    with col_sel:
        selected_model_ui = st.selectbox(
            "Выберите модель для анализа:",
            options=model_options,
            index=0
        )
    
    # Определяем значение для отправки (если выбрана Base -> отправляем None или "Base")
    model_to_send = "Default"
    if selected_model_ui != "Default (Предобученная LoRA)":
        model_to_send = selected_model_ui

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("**Быстрый тест:**")
        examples = {
            "Позитивный": "Хотел поблагодарить за сессию, все прошло отлично!",
            "Нейтральный": "Хотел бы получить копию счета на оплату",
            "Негативный": "Я очень разочарован качеством обучения."
        }
        for label, text_val in examples.items():
            st.button(f"📝 {label}", key=f"btn_{label}", on_click=set_text, args=(text_val,))
    
    with col1:
        text_input = st.text_area(
            "Введите текст:", 
            height=200,
            placeholder="Напишите здесь текст для анализа тональности...",
            key="text_input" 
        )
        
        if st.button("🚀 Анализировать", type="primary", use_container_width=True):
            if text_input:
                with st.spinner(f"Модель '{selected_model_ui}' думает..."):
                    # ПЕРЕДАЕМ model_to_send
                    result = predict_sentiment(text_input, st.session_state.user_id, model_to_send)
                    
                    if result:
                        sentiment_ru, emoji = map_label_to_russian(result['label'])
                        score = result['score']
                        
                        st.markdown("---")
                        st.subheader(f"{emoji} Результат: **{sentiment_ru}**")
                        
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.progress(score)
                            st.metric("Уверенность", f"{score*100:.1f}%")
                        with m_col2:
                            st.caption(f"ID: {result['id']}")
                            st.caption(f"Модель: {selected_model_ui}") 
            else:
                st.warning("✍️ Пожалуйста, введите текст.")

elif page == "Обучение модели":
    st.header("🎓 Дообучение модели (Fine-tuning)")

    training_status = get_training_status()
    available_models = get_available_models()

    if training_status:
        status = training_status.get("status")

        if training_status.get("is_training"):
            st.warning("⏳ Идёт обучение модели")
            st.write(f"Статус: {status}")
            st.write(f"Сообщение: {training_status.get('message')}")
            if st.button("🔄 Обновить статус"):
                st.rerun()
            st.stop()

        if status == "error":
            st.error("❌ Ошибка обучения")
            st.write(training_status.get("message"))

            if st.button("🔄 Сбросить и начать заново"):
                try:
                    requests.post(f"{API_URL}/training/reset", timeout=5)
                except:
                    pass
                st.session_state.dataset_path = None
                st.rerun()

            st.info("Исправьте CSV и загрузите файл заново")

        if status == "completed":
            st.success("✅ Обучение завершено")
            st.write(training_status.get("message"))
            st.info("Можно загрузить новый датасет и запустить обучение снова")

    st.subheader("📂 Загрузка датасета")

    uploaded_file = st.file_uploader(
        "CSV файл с колонками text и label",
        type=["csv"]
    )

    if st.button("📤 Отправить датасет"):
        if uploaded_file is None:
            st.error("Сначала выберите CSV файл")
        else:
            res = upload_dataset(uploaded_file)
            if res:
                st.session_state.dataset_path = res["path"]
                st.success(f"Датасет загружен: {res['rows']} строк")
                st.success(f"Путь: {res['path']}")

    if not st.session_state.dataset_path:
        st.info("⬆️ Загрузите датасет, чтобы продолжить")
    else:
        st.markdown("---")
        st.subheader("⚙️ Параметры обучения")
        st.write(f"📁 Текущий датасет: `{st.session_state.dataset_path}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Настройки данных**")
            epochs = st.number_input("Количество эпох", min_value=1, max_value=20, value=3)
            batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)
            learning_rate = st.selectbox("Learning Rate", [1e-4, 2e-4, 5e-5, 2e-5], index=1)

        with col2:
            st.markdown("**Настройки модели**")
            new_model_name = st.text_input(
                "Имя новой модели (для сохранения)", 
                value=f"model_{int(time.time())}"
            )

            source_options = ["Базовая модель (RuBERT)"] + available_models
            selected_source_ui = st.selectbox(
                "Начать обучение на основе:",
                options=source_options
            )

        source_path_to_send = None
        if selected_source_ui != "Базовая модель (RuBERT)":
            source_path_to_send = f"./trained_models/{selected_source_ui}"
            st.info(f"Будет выполнено дообучение модели: {selected_source_ui}")
        else:
            st.info("Будет использована базовая модель RuBERT с инициализацией новых весов.")

        st.markdown("---")

        if st.button("🔥 Начать обучение", type="primary", use_container_width=True):
            if not new_model_name.strip():
                st.warning("Пожалуйста, укажите имя для сохранения модели")
            else:
                res = start_training(
                    st.session_state.dataset_path,
                    epochs,
                    batch_size,
                    learning_rate,
                    new_model_name,
                    source_path_to_send
                )
                if res:
                    st.success("Обучение успешно запущено")
                    time.sleep(1)
                    st.rerun()

elif page == "История обучения":
    st.header("📜 Лог обучений")
    data = get_training_history()
    
    if data and data.get("history"):
        df = pd.DataFrame(data["history"])
        
        # Улучшаем отображение таблицы
        # Выбираем и переименовываем колонки для красоты
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime('%Y-%m-%d %H:%M')
        
        # Порядок колонок
        cols_to_show = ["model_name", "timestamp", "num_epochs", "train_loss", "train_samples"]
        
        # Если каких-то колонок нет (например, старые записи), не падаем
        available_cols = [c for c in cols_to_show if c in df.columns]
        
        st.dataframe(
            df[available_cols].rename(columns={
                "model_name": "Имя модели",
                "timestamp": "Дата",
                "num_epochs": "Эпохи",
                "train_loss": "Ошибка (Loss)",
                "train_samples": "Размер данных"
            }),
            use_container_width=True
        )
    else:
        st.info("История пуста.")

elif page == "История предсказаний":
    st.header("🗂 Мои запросы")
    history = get_user_history(st.session_state.user_id)
    
    if history:
        df = pd.DataFrame(history)
        df['Тональность'] = df['label'].apply(lambda x: map_label_to_russian(x)[1] + " " + map_label_to_russian(x)[0])
        df['Уверенность'] = df['score'].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(
            df[['created_at', 'text', 'Тональность', 'Уверенность']], 
            use_container_width=True
        )
    else:
        st.info("Вы пока ничего не анализировали.")

st.markdown("---")
st.markdown("<center><small>Разработано в рамках практики | 2025</small></center>", unsafe_allow_html=True)