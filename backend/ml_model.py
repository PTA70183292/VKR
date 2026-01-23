import torch     # Импорт библиотеки PyTorch для вычислений и работы с нейросетями.
import os        # Импорт модуля OS для доступа к переменным окружения.
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig 
# Импорт классов для работы с моделями Hugging Face.

from peft import PeftModel         # Импорт класса для работы с LoRA/PEFT адаптерами.
from config import settings        # Импорт конфигурационных параметров из файла config.py.

# Логика определения устройства для вычислений
REQUESTED_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu").lower()         # Читаем желаемый режим из .env (по умолчанию 'cpu').

# Если просили GPU, но драйверов нет - переключаемся на CPU, иначе используем запрошенное
DEVICE = "cuda" if (REQUESTED_DEVICE == "gpu" and torch.cuda.is_available()) else "cpu" # Итоговое устройство выполнения (cuda или cpu).

print(f"Режим работы ML-модели: {DEVICE.upper()}")                      # Выводим в лог текущее устройство вычислений.

class SentimentModel:                                                   # Основной класс-обертка для модели анализа тональности.
    def __init__(self):                                                 # Конструктор класса, инициализирующий модель при старте.
        print("Инициализация модели...")                                # Логирование начала процесса загрузки.
        
        # 1. Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name) # Загрузка токенизатора для преобразования текста в токены.

        # 2. Логика загрузки БАЗОВОЙ модели (Зависит от устройства)
        print(f"Загрузка базовой модели: {settings.base_model_name}...") # Логирование загрузки весов базовой модели.
        
        if DEVICE == "cuda":                                            # Ветка выполнения для GPU (NVIDIA).
            # ПУТЬ GPU (Быстрый, с квантованием 8-bit)
            try:                                                        # Блок обработки ошибок инициализации CUDA/BitsAndBytes.
                bnb_config = BitsAndBytesConfig(                        # Настройка конфигурации квантования.
                    load_in_8bit=True,                                  # Включение 8-битного режима для экономии видеопамяти.
                    llm_int8_threshold=6.0                              # Порог отсечения для int8 конвертации.
                )
                self.base_model = AutoModelForSequenceClassification.from_pretrained( # Загрузка архитектуры классификатора.
                    settings.base_model_name,                           # Имя модели из конфигурации.
                    num_labels=3,                                       # Количество классов классификации.
                    quantization_config=bnb_config,                     # Применение конфига квантования.
                    device_map="auto"                                   # Автоматическое распределение слоев по GPU.
                )
                print("Базовая модель загружена на GPU (8-bit)")        
            except Exception as e:                                      # Перехват ошибки (например, старый драйвер).
                print(f"Ошибка GPU, переходим на CPU: {e}")             # Логирование ошибки.
                self._load_base_cpu()                                   # Аварийное переключение на загрузку через CPU.
        else:                                                           # Ветка выполнения для CPU (VPS/Слабый ПК).
            # ПУТЬ CPU (Медленный, точность FP32)
            self._load_base_cpu()                                       # Вызов метода загрузки для процессора.

        # 3. Инициализируем PEFT с ДЕФОЛТНЫМ адаптером
        print(f"Загрузка дефолтного адаптера: {settings.adapter_name}...")
        try:                                                            # Попытка загрузить веса адаптера.
            self.model = PeftModel.from_pretrained(                     # Оборачиваем базовую модель в PEFT.
                self.base_model,                                        # Базовая модель-скелет.
                settings.adapter_name,                                  # Путь к весам адаптера.
                adapter_name="default",                                 # Внутреннее имя адаптера в памяти.
                is_trainable=False                                      # Запрещаем обучение (только инференс).
            )
            print("Дефолтный адаптер загружен и активен.")              # Подтверждение успеха.
        except Exception as e:                                          # Если адаптер не найден или поврежден.
            print(f"Ошибка загрузки адаптера (работаем на чистой базе): {e}") # Предупреждение.
            self.model = self.base_model                                # Используем "голую" модель без дообучения.

        # Дополнительная проверка для CPU
        if DEVICE == "cpu":                                             # Если работаем на процессоре.
            self.model.to("cpu")                                        # Явно перемещаем веса в оперативную память (RAM).

        self.active_adapter_name = "default"                            # Сохраняем имя текущего активного адаптера.

    def _load_base_cpu(self):                                           # Вспомогательный метод для инициализации на CPU.
         #Вспомогательная функция для загрузки на процессор без квантования
        self.base_model = AutoModelForSequenceClassification.from_pretrained( # Загрузка стандартной модели.
            settings.base_model_name,                                   # Имя модели.
            num_labels=3,                                               # Количество классов.
            ignore_mismatched_sizes=True                                # Игнорировать несовпадение размеров слоев (если меняли голову).
        )
        self.base_model.to("cpu")                                       # Перенос модели на CPU.
        print("Базовая модель загружена на CPU (FP32)")                 # Логирование успеха.

    def switch_model(self, model_name: str):                            # Метод для динамического переключения адаптеров.
        
         #Переключает активный адаптер без перезагрузки всей модели.
        
        target_adapter = "default"                                      # Установка целевого адаптера по умолчанию.
        if model_name and model_name not in ["QLoRA r64", "Default", "Base", "default"]: # Проверка, запрошено ли специфичное имя.
            target_adapter = model_name                                 # Установка запрошенного имени.

        if self.active_adapter_name == target_adapter:                  # Если адаптер уже активен.
            return                                                      # Ничего не делаем, выходим.

        print(f"Переключение адаптера на '{target_adapter}'...")        # Логирование процесса переключения.

        # Проверяем, является ли модель экземпляром PeftModel
        if not isinstance(self.model, PeftModel):                       # Если модель еще не обернута (голая база).
            adapter_path = f"./trained_models/{target_adapter}"         # Формирование пути к файлам адаптера.
            if not os.path.exists(adapter_path):                        # Проверка существования папки.
                return                                                  # Папки нет - выходим.
            try:                                                        # Попытка первой инициализации PEFT.
                self.model = PeftModel.from_pretrained(                 # Создаем PEFT-обертку.
                    self.base_model, adapter_path, adapter_name=target_adapter
                )
                if DEVICE == "cpu": self.model.to("cpu")                # Для CPU перемещаем модель в RAM.
                self.active_adapter_name = target_adapter               # Обновляем имя активного адаптера.
                return                                                  # Успешный выход.
            except Exception:                                           # При ошибке загрузки.
                return                                                  # Выходим без изменений.

        # Если это уже PeftModel (сценарий переключения)
        if target_adapter in self.model.peft_config:                    # Проверяем, загружен ли адаптер в кэш памяти.
            try:                                                        # Попытка горячего переключения.
                self.model.set_adapter(target_adapter)                  # Активируем нужный слой весов.
                self.active_adapter_name = target_adapter               # Обновляем состояние.
            except Exception as e:                                      # Ошибка переключения.
                print(f"Ошибка переключения: {e}")                      # Лог ошибки.
        else:                                                           # Адаптера нет в памяти, нужно грузить с диска.
            adapter_path = f"./trained_models/{target_adapter}"         # Путь к файлам.
            if not os.path.exists(adapter_path):                        # Если файла нет.
                if "default" in self.model.peft_config:                 # Пробуем откатиться на дефолт.
                    self.model.set_adapter("default")                   # Активируем дефолт.
                    self.active_adapter_name = "default"                # Обновляем состояние.
                return                                                  # Выходим.

            try:                                                        # Попытка загрузки нового адаптера.
                self.model.load_adapter(adapter_path, adapter_name=target_adapter) # Загружаем веса в память.
                self.model.set_adapter(target_adapter)                  # Делаем его активным.
                if DEVICE == "cpu": self.model.to("cpu")                # Гарантируем нахождение на CPU.
                self.active_adapter_name = target_adapter               # Обновляем состояние.
                print(f"Загружена: {target_adapter}")                   # Лог успеха.
            except Exception as e:                                      # Ошибка I/O или формата.
                print(f"Ошибка загрузки адаптера {target_adapter}: {e}") # Лог ошибки.

    def predict(self, text: str, model_name: str = None) -> dict:       # Метод для выполнения инференса (предсказания).
        self.switch_model(model_name)                                   # Сначала применяем нужный адаптер.

        if not text:                                                    # Проверка на пустой ввод.
            return {"label": "neutral", "score": 0.0}                   # Возврат заглушки для пустого текста.

        inputs = self.tokenizer(                                        # Токенизация входного текста.
            text,                                                       # Исходная строка.
            return_tensors="pt",                                        # Возврат тензоров PyTorch.
            truncation=True,                                            # Обрезка длинных текстов.
            padding=True,                                               # Добавление паддингов (если батч).
            max_length=512                                              # Максимальная длина контекста модели.
        )
        
        # Перенос данных на устройство (автоматически cpu или cuda)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()} # Копируем тензоры на то же устройство, где модель.

        with torch.no_grad():                                           # Отключаем расчет градиентов (экономит память).
            outputs = self.model(**inputs)                              # Прогон данных через нейросеть (прямой проход).
            probs = torch.softmax(outputs.logits, dim=1)                # Преобразование логитов в вероятности (Softmax).

        score, label_id = torch.max(probs, dim=1)                       # Поиск класса с максимальной вероятностью.

        return {                                                        # Формирование словаря с ответом.
            "label": f"LABEL_{label_id.item()}",                        # Метка класса (например, LABEL_0).
            "score": float(score.item())                                # Уверенность модели (float).
        }

# Паттерн Singleton для глобального доступа к модели
sentiment_model = None                                                  # Глобальная переменная для хранения экземпляра.

def get_sentiment_model() -> SentimentModel:                            # Функция-аксессор для получения модели.
    global sentiment_model                                              # Обращаемся к глобальной переменной.
    if sentiment_model is None:                                         # Если модель еще не создана.
        sentiment_model = SentimentModel()                              # Создаем экземпляр (тяжелая операция).
    return sentiment_model                                              # Возвращаем готовый объект.
