import streamlit as st  # Основная библиотека для UI
import requests  # Для общения с Backend API
import pandas as pd  # Для отображения таблиц и обработки данных
import time  # Для задержек (например, чтобы пользователь успел прочитать сообщение об успехе)
import os  # Для чтения переменных окружения
from datetime import datetime  # Для работы с датами


# Настройка заголовка вкладки и макета страницы
st.set_page_config(
    page_title="Обращения студентов", 
    layout="wide", 
    page_icon="👤"
)

# Инициализация переменных сессии (state), если они еще не созданы
# Это нужно, чтобы данные не терялись при обновлении страницы (действии пользователя)
if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = None
if "dataset_uploaded" not in st.session_state:
    st.session_state.dataset_uploaded = False
    
# Состояние авторизации
if "token" not in st.session_state:
    st.session_state.token = None  # JWT токен
if "role" not in st.session_state:
    st.session_state.role = None   # Роль пользователя (admin/user)
if "username" not in st.session_state:
    st.session_state.username = None

# Состояние просмотра тикетов (список или детали конкретного тикета)
if "ticket_view_mode" not in st.session_state: st.session_state.ticket_view_mode = "list"
if "selected_ticket_id" not in st.session_state: st.session_state.selected_ticket_id = None

# Получаем адрес API из Docker-переменных или используем локальный адрес по умолчанию
API_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# Функции авторизации
def get_headers():
    #Добавляет JWT-токен в заголовки запроса, если пользователь авторизован.
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

def get_active_model_api():
    #Запрашивает у бэкенда имя текущей активной модели.
    try:
        r = requests.get(f"{API_URL}/config/active-model", headers=get_headers())
        return r.json().get("model_name", "QLoRA r64")
    except:
        return "QLoRA r64"  # Фоллбэк, если API недоступен

def set_active_model_api(name):
    #Отправляет запрос на смену активной модели.
    try:
        r = requests.post(f"{API_URL}/config/active-model", json={"model_name": name}, headers=get_headers())
        return r.status_code == 200
    except:
        return False

def login(username, password):
    #Попытка входа в систему.
    try:
        # Используем form-data, так как OAuth2PasswordRequestForm на бэкенде ожидает именно его
        resp = requests.post(
            f"{API_URL}/token", 
            data={"username": username, "password": password},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            # Сохраняем данные пользователя в сессию
            st.session_state.token = data["access_token"]
            st.session_state.role = data["role"]
            st.session_state.username = data["username"]
            st.rerun()  # Перезагружаем приложение, чтобы отобразить интерфейс вместо формы входа
        else:
            st.error("Неверный логин или пароль")
    except Exception as e:
        st.error(f"Ошибка подключения: {e}")

def upload_model_zip(file):
    #Загрузка ZIP-архива с моделью.
    try:
        files = {"file": (file.name, file, "application/zip")}
        # Увеличенный таймаут (60с), так как файлы могут быть большими
        r = requests.post(f"{API_URL}/training/upload-model-zip", files=files, headers=get_headers(), timeout=60)
        
        if r.status_code == 200:
            return True, r.json().get("message", "ОК")
        else:
            # Пытаемся достать детальное описание ошибки из ответа
            try:
                err = r.json().get("detail", r.text)
            except:
                err = r.text
            return False, err
    except Exception as e:
        return False, str(e)


# Функции-обертки для API)
def get_tickets():
    #Получение списка всех тикетов (фильтрация по роли происходит на бэкенде).
    try:
        r = requests.get(f"{API_URL}/tickets", headers=get_headers(), timeout=5)
        return r.json() if r.status_code == 200 else []
    except:
        return []

def get_ticket_details(t_id):
    #Получение одного тикета по ID.
    try:
        r = requests.get(f"{API_URL}/tickets/{t_id}", headers=get_headers(), timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def update_ticket_label_api(t_id, new_label):
    #Ручное обновление метки (тональности) тикета.
    try:
        payload = {"label": new_label}
        r = requests.put(f"{API_URL}/tickets/{t_id}/label", json=payload, headers=get_headers())
        return r.status_code == 200
    except:
        return False

def map_label_visual(label):
    #Преобразует техническую метку (LABEL_1) в человекочитаемый вид с смайликом.
    d = {
        "LABEL_0": ("Нейтрально", "😐"), 
        "LABEL_1": ("Позитивно", "😁"), 
        "LABEL_2": ("Негативно", "😡"),
        # Дублирование ключей для надежности (если модель вернет текстовое название)
        "neutral": ("Нейтрально", "😐"), 
        "positive": ("Позитивно", "😁"), 
        "negative": ("Негативно", "😡"),
        "⏳ Анализ...": ("Анализ...", "⏳")
    }
    # Возвращает кортеж (Текст, Смайлик). Если метка не найдена - возвращает саму метку и знак вопроса.
    return d.get(label, (label, "❓"))

def get_label_code_by_name(russian_name):
    #Обратный маппинг: Русское название -> Техническая метка (для отправки на сервер).
    mapping = {
        "Нейтрально": "LABEL_0",
        "Позитивно": "LABEL_1",
        "Негативно": "LABEL_2"
    }
    return mapping.get(russian_name, "LABEL_0")

# Функция раскраски таблицы (Pandas Styler)
def highlight_sentiment(val):
    #Возвращает CSS-стили для ячейки таблицы в зависимости от значения тональности.
    val_str = str(val).lower()
    if 'негативно' in val_str:
        return 'background-color: #ffcdd2; color: #b71c1c'  # Светло-красный фон
    elif 'позитивно' in val_str:
        return 'background-color: #c8e6c9; color: #1b5e20'  # Светло-зеленый фон
    elif 'анализ' in val_str:
        return 'background-color: #fff9c4; color: #f57f17'  # Желтый (ожидание)
    return ''

def check_api_health():
    #Проверка доступности бэкенда.
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def upload_dataset(file):
    #Загрузка CSV датасета для обучения.
    try:
        files = {"file": (file.name, file, "text/csv")}
        response = requests.post(
            f"{API_URL}/training/upload-dataset",
            files=files,
            headers=get_headers(),
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        
        # Обработка ошибок валидации файла (например, неверные колонки)
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
    #Получает список доступных обученных моделей (папок).
    try:
        response = requests.get(
            f"{API_URL}/training/models-list", 
            headers=get_headers(),
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except:
        return []

def start_training(dataset_path, num_epochs, batch_size, learning_rate, custom_model_name, source_model_path):
    #Запуск процесса обучения.
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
            headers=get_headers(),
            timeout=120  # Долгий таймаут, так как инициализация обучения может занять время
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
    #Получение текущего статуса обучения (прогресс, этап).
    try:
        response = requests.get(
            f"{API_URL}/training/status", 
            headers=get_headers(),
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_training_history():
    #Получение истории всех проведенных обучений.#
    try:
        response = requests.get(
            f"{API_URL}/training/history", 
            headers=get_headers(),
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except:
        return {"history": []}


# Управление пользователями)

def get_users_list():
    try:
        r = requests.get(f"{API_URL}/users", headers=get_headers(), timeout=5)
        return r.json() if r.status_code == 200 else []
    except:
        return []

def create_new_user(username, password, role):
    try:
        r = requests.post(
            f"{API_URL}/users", 
            json={"username": username, "password": password, "role": role},
            headers=get_headers(), timeout=5
        )
        return r.status_code == 200, r.text
    except Exception as e:
        return False, str(e)

def delete_user_by_username(username):
    try:
        r = requests.delete(f"{API_URL}/users/username/{username}", headers=get_headers(), timeout=5)
        if r.status_code == 200:
            return True, "OK"
        else:
            try:
                err_msg = r.json().get("detail", r.text)
            except:
                err_msg = r.text
            return False, err_msg
    except Exception as e:
        return False, str(e)


# 1. Экран входа

# Если токена нет - показываем ТОЛЬКО форму входа
if not st.session_state.token:
    st.markdown("<h1 style='text-align: center;'>Обращения студентов</h1>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("auth_form"):
            user_login = st.text_input("Почта")
            user_pass = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", type="primary", use_container_width=True)
            
            if submitted:
                if not user_login or not user_pass:
                    st.warning("Введите логин и пароль")
                else:
                    login(user_login, user_pass)
    
    st.markdown("---")
    st.info("Разработано в рамках ВКР. Студент (о.ИЗДт 23.2/Б3-21) (70183292)")
    st.stop()  # Останавливаем выполнение скрипта, чтобы не показывать интерфейс ниже


# 2. Основной интерфейс
# Проверяем доступность бэкенда
api_status = check_api_health()

# Сайдбар (Боковая панель)
with st.sidebar:
    st.header("Профиль")
    st.write(f"**{st.session_state.username}**")
    
    role_map = {
        "admin": "Администратор",
        "manager": "Руководитель",
        "user": "Пользователь"
    }
    st.caption(role_map.get(st.session_state.role, st.session_state.role))
    
    if st.button("Выйти"):
        # Сброс сессии при выходе
        st.session_state.token = None
        st.session_state.role = None
        st.session_state.username = None
        st.rerun()
    
    st.markdown("---")
    st.header("Меню")

    #RBAC (Role-Based Access Control) Меню
    menu_options = ["Список обращений"]
    
    # Дополнительные пункты меню только для админа
    if st.session_state.role == "admin":
        menu_options.append("Модели и Обучение")
        menu_options.append("История обучения")
        menu_options.append("Управление пользователями")

    page = st.radio("Разделы:", menu_options, key="page_selector")
    
    st.markdown("---")
    if api_status:
        st.success("API: Подключено")
    else:
        st.error("API: Недоступно")

# Если бэкенд лежит - не даем работать дальше
if not api_status:
    st.warning("Не удается подключиться к Бэкенду.")
    st.stop()



# СПИСОК ОБРАЩЕНИЙ 
if page == "Список обращений":
    
    # РЕЖИМ 1: СПИСОК (ТАБЛИЦА)
    if st.session_state.ticket_view_mode == "list":
        st.header("Входящие обращения")
        
        tickets = get_tickets()
        if not tickets:
            st.info("Список пуст.")
        else:
            df = pd.DataFrame(tickets)
            
            # ПРЕПРОЦЕССИНГ
            # 1. Формируем строку "Смайлик Текст" для красивого отображения в таблице
            df["sent_full"] = df["label"].apply(lambda x: f"{map_label_visual(x)[1]} {map_label_visual(x)[0]}")
            
            # 2. Обработка дат для фильтрации
            df["created_dt"] = pd.to_datetime(df["created_at"]) 
            df["created_at_str"] = df["created_dt"].dt.strftime('%d.%m.%Y %H:%M')
            
            # 3. Заполнение пропусков (NaN)
            if "assigned_to" not in df.columns: df["assigned_to"] = "Не назначен"
            else: df["assigned_to"] = df["assigned_to"].fillna("Не назначен")
            
            if "status" not in df.columns: df["status"] = "Новое"

            # ФИЛЬТРЫ (В блоке Expander)
            with st.expander("Фильтры и Поиск", expanded=True):
                col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
                
                with col_f1:
                    filter_sentiment = st.multiselect(
                        "Тональность", 
                        ["Нейтрально", "Позитивно", "Негативно","Анализ..."],
                        default=[]
                    )
                with col_f2:
                    # Фильтр по датам
                    today = datetime.now().date()
                    start_default = today.replace(day=1) # Начало месяца
                    
                    date_range = st.date_input(
                        "Период (От - До)",
                        value=(start_default, today),
                        format="DD.MM.YYYY"
                    )
                with col_f3:
                    search_text = st.text_input("Поиск (Тема)")

            # ПРИМЕНЕНИЕ ФИЛЬТРОВ
            filtered_df = df.copy()
            
            # 1. По тональности
            if filter_sentiment:
                pattern = '|'.join(filter_sentiment)
                filtered_df = filtered_df[filtered_df["sent_full"].str.contains(pattern, case=False)]
            
            # 2. По дате
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_d, end_d = date_range
                mask = (filtered_df["created_dt"].dt.date >= start_d) & (filtered_df["created_dt"].dt.date <= end_d)
                filtered_df = filtered_df[mask]
            
            # 3. Поиск по теме
            if search_text:
                filtered_df = filtered_df[filtered_df["subject"].str.contains(search_text, case=False)]

             #ЭКСПОРТ В CSV
            st.markdown("###") 
            col_res, col_exp = st.columns([6, 2])
            with col_res:
                st.write(f"Найдено записей: **{len(filtered_df)}**")
            
            with col_exp:
                # Подготовка данных для отчета
                filtered_df["label_text"] = filtered_df["label"].apply(lambda x: map_label_visual(x)[0])

                export_cols = ["id", "created_at_str", "user_email", "assigned_to", "subject", "description", "status", "label_text", "label"]
                available_cols = [c for c in export_cols if c in filtered_df.columns]
                
                export_df = filtered_df[available_cols].rename(columns={
                    "id": "ID", "created_at_str": "Дата", "user_email": "Студент", "assigned_to": "Исполнитель",
                    "subject": "Тема", "text": "Текст обращения", "status": "Статус",
                    "label_text": "Тональность (Текст)", "label": "Тональность (Код)"
                })
                
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать отчет (.csv)",
                    data=csv,
                    file_name=f"dataset_export_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )

            # ОТОБРАЖЕНИЕ ТАБЛИЦЫ
            display_df = filtered_df[["id", "created_at_str", "user_email", "assigned_to", "subject", "sent_full"]].rename(columns={
                "id": "ID", "created_at_str": "Дата", "user_email": "Отправитель", 
                "assigned_to": "Исполнитель", "subject": "Тема", "sent_full": "Тон" 
            })
            # Применяем раскраску ячеек
            styled_df = display_df.style.map(highlight_sentiment, subset=['Тон'])
            st.info("Нажмите на галочку для просмотра деталей")
            
            # Интерактивная таблица (on_select="rerun" позволяет обработать клик по строке)
            event = st.dataframe(styled_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
            if len(event.selection.rows) > 0:
                selected_index = event.selection.rows[0]
                ticket_id = display_df.iloc[selected_index]["ID"]
                st.session_state.selected_ticket_id = ticket_id
                st.session_state.ticket_view_mode = "detail" # Переключаем режим на детализацию
                st.rerun()

    # РЕЖИМ 2: ДЕТАЛИ
    elif st.session_state.ticket_view_mode == "detail":
        t_id = st.session_state.selected_ticket_id
        if st.button("⬅️ Назад к списку"):
            st.session_state.ticket_view_mode = "list"
            st.session_state.selected_ticket_id = None
            st.rerun()
            
        detail = get_ticket_details(t_id)
        if detail:
            st.title(detail['subject'])
            st.caption(f"ID обращения: #{detail['id']}")
            
            with st.container(border=True):
                c1, c2 = st.columns(2)
                c1.markdown(f"**От:** `{detail['user_email']}`")
                c1.markdown(f"**Дата:** {detail['created_at']}")
                assignee_show = detail.get('assigned_to') or "Не назначен"
                c2.markdown(f"**Исполнитель:** `{assignee_show}`")
                status_txt = detail.get('status', 'Новое')
                status_color = "green" if status_txt == "Закрыто" else "blue"
                c2.markdown(f"**Статус:** :{status_color}[{status_txt}]")
                st.markdown("---")
                st.markdown("**Текст обращения:**")
                st.info(detail.get('description', 'Текст отсутствует'))
            
            st.markdown("###")
            st.markdown("#### Анализ тональности ИИ:")
            
            col_res, col_fix = st.columns([2, 1])
            
            # ЛЕВАЯ КОЛОНКА: Результат модели
            with col_res:
                sent_text, emoji = map_label_visual(detail['label'])
                score_pct = detail['score'] * 100
                lbl = str(detail['label']).lower()

                if detail.get('model_name') == "Manual":
                    score_display = "(вручную)"
                else:
                    score_display = f"({score_pct:.1f}%)"

                # Вывод плашки с цветом в зависимости от тональности
                if "negative" in lbl or "label_2" in lbl:
                        st.error(f"## 😡 Тон: {sent_text} {score_display}")
                elif "positive" in lbl or "label_1" in lbl:
                        st.success(f"## 😁 Тон: {sent_text} {score_display}")
                else:
                        st.info(f"## 😐 Тон: {sent_text} {score_display}")
                
                st.caption(f"Модель: `{detail['model_name']}`")

            # ПРАВАЯ КОЛОНКА: Ручная правка (Human-in-the-loop)
            with col_fix:
                with st.container(border=True):
                    st.write("**Ошибка ИИ?**")
                    new_sentiment_ru = st.selectbox(
                        "Исправить на:",
                        ["Позитивно", "Нейтрально", "Негативно"],
                        index=None,
                        placeholder="Выберите...",
                        label_visibility="collapsed"
                    )
                    
                    if new_sentiment_ru:
                        if st.button("💾 Сохранить", type="secondary", use_container_width=True):
                            new_code = get_label_code_by_name(new_sentiment_ru)
                            if update_ticket_label_api(t_id, new_code):
                                st.success("Обновлено!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error("Ошибка сохранения")

        else:
            st.error("Ошибка загрузки тикета")
            if st.button("Вернуться"):
                st.session_state.ticket_view_mode = "list"
                st.rerun()


elif page == "Модели и Обучение":
    st.header("Управление ИИ-моделями")

    # БЛОК 1: ВЫБОР АКТИВНОЙ МОДЕЛИ
    st.subheader("Глобальная модель")
    st.info("Выберите модель, которая будет анализировать новые обращения студентов.")
    
    available = ["QLoRA r64"] + get_available_models()
    current_active = get_active_model_api()
    
    try:
        curr_index = available.index(current_active)
    except:
        curr_index = 0
        
    c_sel, c_save = st.columns([3, 1])
    with c_sel:
        selected_for_prod = st.selectbox(
            "Активная модель:", 
            options=available, 
            index=curr_index,
            key="prod_model_selector"
        )
    with c_save:
        st.write("") 
        st.write("")
        if st.button("Применить", type="primary", use_container_width=True):
            if set_active_model_api(selected_for_prod):
                st.success(f"Модель '{selected_for_prod}' установлена!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Ошибка сохранения")

    st.markdown("---")

    # БЛОК 1.5: ЗАГРУЗКА ГОТОВОЙ МОДЕЛИ (ZIP
    with st.expander("Загрузить готовую модель (адаптер) из файла (zip)"):
        st.info("Загрузите архив, содержащий файлы адаптера (adapter_config.json, adapter_model.bin). Имя модели будет взято из названия файла.")
        
        uploaded_zip = st.file_uploader("Выберите ZIP-архив", type="zip", key="model_uploader")
        
        if uploaded_zip:
            if st.button("Загрузить модель на сервер", type="primary"):
                with st.spinner("Загрузка и распаковка..."):
                    ok, msg = upload_model_zip(uploaded_zip)
                    if ok:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun() 
                    else:
                        st.error(f"Ошибка: {msg}")

    st.markdown("---")

    
    # БЛОК 2: ОБУЧЕНИЕ
    st.header("Дообучение (Fine-Tuning) \n *(Функция работает при запуске локально с GPU. Через хостинг проект реализован только в режиме инференса.)*")

    # 1. ПРОВЕРКА ТЕКУЩЕГО СТАТУСА
    training_status = get_training_status()
    
    # Если обучение идет, блокируем форму запуска и показываем статус
    if training_status and (training_status.get("is_training") or training_status.get("status") == "error"):
        
        status_msg = training_status.get("message", "")
        
        if training_status.get("status") == "error":
            st.error(f"Ошибка обучения: {status_msg}")
        else:
            st.info(f"Статус процесса: {status_msg}")
            
        col_refresh, col_cancel = st.columns(2)
        
        with col_refresh:
            if st.button("Обновить статус"):
                st.rerun()
                
        st.stop() # Не рисуем форму запуска

    # 2. ФОРМА ЗАПУСКА
    st.write("Загрузите CSV (text, label) для создания новой модели:")
    
    uploaded_file = st.file_uploader("CSV файл (text, label)", type=["csv"])
    if st.button("Загрузить датасет"):
        if uploaded_file:
            res = upload_dataset(uploaded_file)
            if res:
                st.session_state.dataset_path = res["path"]
                st.success(f"Загружено строк: {res['rows']}")

    if st.session_state.dataset_path:
        st.write("Параметры обучения:")
        c1, c2 = st.columns(2)
        with c1:
            ep = st.number_input("Эпохи", 1, 10, 3)
            lr = st.selectbox("Learning Rate", [2e-5, 5e-5], index=0)
        with c2:
            # Генерация уникального имени модели по умолчанию (только один раз за сессию, чтобы не прыгало)
            if "default_model_name" not in st.session_state:
                st.session_state.default_model_name = f"model_{int(time.time())}"
            
            new_m_name = st.text_input("Имя новой модели", value=st.session_state.default_model_name)
            
            src = st.selectbox("База:", ["QLoRA r64"] + get_available_models())
            
        if st.button("Начать обучение"):
             # Сбрасываем дефолтное имя, чтобы при следующем обучении оно сгенерировалось заново
             st.session_state.pop("default_model_name", None)
             
             src_path = f"./trained_models/{src}" if src != "QLoRA r64" else None
             res = start_training(st.session_state.dataset_path, ep, 8, lr, new_m_name, src_path)
             if res: 
                 st.success("Запущено!")
                 time.sleep(1)
                 st.rerun()


#ИСТОРИЯ ОБУЧЕНИЯ
elif page == "История обучения":
    st.header("Лог обучений")
    data = get_training_history()
    
    if data and data.get("history"):
        df = pd.DataFrame(data["history"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime('%Y-%m-%d %H:%M')
        
        cols = ["model_name", "timestamp", "num_epochs", "train_loss", "train_samples"]
        av_cols = [c for c in cols if c in df.columns]
        
        st.dataframe(df[av_cols].rename(columns={"model_name":"Имя", "timestamp":"Дата", "train_loss":"Loss"}), use_container_width=True)
    else:
        st.info("История пуста.")

#УПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯМИ
elif page == "Управление пользователями":
    st.header("👥 Сотрудники и Доступы")
    
    # 1. Список (получаем данные)
    users = get_users_list()
    
    if users:
        df_u = pd.DataFrame(users)
        
        # Показываем только нужные колонки (без ID и хэша пароля)
        columns_to_show = ['username', 'role', 'is_active']
        st.dataframe(
            df_u[columns_to_show].rename(columns={"username":"Логин", "role":"Роль"}),
            use_container_width=True
        )
    else:
        st.info("Нет пользователей или ошибка загрузки")
    
    st.markdown("---")
    c_add, c_del = st.columns(2)
    
    # 2. Добавление пользователя
    with c_add:
        st.subheader("Добавить пользователя")
        with st.form("new_user_form"):
            nu_login = st.text_input("Новый Логин (Email)")
            nu_pass = st.text_input("Пароль", type="password")
            nu_role = st.selectbox("Роль", ["user", "manager", "admin"])
            if st.form_submit_button("Создать"):
                ok, msg = create_new_user(nu_login, nu_pass, nu_role)
                if ok:
                    st.success("Пользователь создан")
                    st.rerun()
                else:
                    st.error(f"Ошибка: {msg}")

    # 3. Удаление пользователя
    with c_del:
        st.subheader("Удалить пользователя")
        if users:
            # Исключаем текущего пользователя из списка (чтобы админ не удалил сам себя)
            usernames_list = [u['username'] for u in users if u['username'] != st.session_state.username]
            
            selected_user_to_del = st.selectbox("Выберите пользователя", usernames_list)
            
            if st.button("🗑 Удалить выбранного"):
                if selected_user_to_del:
                    ok, msg = delete_user_by_username(selected_user_to_del)
                    if ok:
                        st.success(f"Пользователь {selected_user_to_del} удален")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Ошибка: {msg}")
        else:
            st.caption("Список пуст")

st.markdown("---")
st.markdown("Разработано в рамках ВКР. Студент (о.ИЗДт 23.2/Б3-21) (70183292)", unsafe_allow_html=True)