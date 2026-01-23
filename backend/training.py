import os                                                # Модуль для работы с файловой системой (пути, папки)
from transformers import (                               # Импорт компонентов Hugging Face Transformers
    AutoTokenizer,                                       # Класс для токенизации текста
    AutoModelForSequenceClassification,                  # Базовый класс модели для задачи классификации
    TrainingArguments,                                   # Класс настройки параметров обучения
    Trainer,                                             # Главный класс, управляющий циклом обучения
    DataCollatorWithPadding                              # Утилита для выравнивания длины батчей (padding)
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel # Библиотека PEFT для LoRA адаптеров
from datasets import Dataset                             # Библиотека для эффективной работы с данными
import pandas as pd                                      # Библиотека для работы с таблицами (CSV)
from typing import Dict, Tuple                           # Типизация для чистоты кода
from config import settings                              # Импорт глобальных настроек проекта
import json                                              # Модуль для сохранения истории/логов в JSON
from datetime import datetime                            # Модуль времени
import shutil                                            # Модуль для операций с файлами (удаление папок)

class SentimentTrainer:                                  # Класс-менеджер процесса обучения
    def __init__(self, base_model_name: str = None):     # Конструктор
        self.base_model_name = base_model_name or settings.base_model_name # Имя модели (из аргумента или конфига)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name) # Загружаем токенизатор под эту модель
        self.model = None                                # Слот для самой модели
        self.training_history = []                       # Список для хранения логов обучения
    
    def load_dataset_from_csv(self, file_path: str) -> Dataset: # Загрузка и валидация CSV
        df = pd.read_csv(file_path)                      # Читаем CSV в Pandas DataFrame
        required_columns = ['text', 'label']             # Обязательные колонки
        if not all(col in df.columns for col in required_columns): # Проверка структуры
            raise ValueError(f"CSV должен содержать колонки: {required_columns}")
        
        if df['label'].dtype == 'object':                # Если метки пришли текстом (напр. "positive")
            label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())} # Создаем карту 'pos'->1
            df['label'] = df['label'].map(label_mapping) # Заменяем текст на цифры
        
        dataset = Dataset.from_pandas(df[['text', 'label']]) # Конвертируем в формат Hugging Face Dataset
        return dataset

    def preprocess_function(self, examples):             # Вспомогательная функция токенизации
        return self.tokenizer(
            examples['text'],
            truncation=True,                             # Обрезаем, если текст длиннее максимума
            padding=True,                                # Добиваем нулями до длины самого длинного в батче
            max_length=128                               # Максимальная длина последовательности
        )
    
    def prepare_dataset(self, dataset: Dataset, test_size: float = 0.2) -> Tuple[Dataset, Dataset]: # Разделение выборки
        tokenized_dataset = dataset.map(                 # Применяем токенизацию ко всему датасету
            self.preprocess_function,
            batched=True,
            remove_columns=['text']                      # Удаляем сырой текст, оставляем токены
        )
        split_dataset = tokenized_dataset.train_test_split(test_size=test_size) # Делим на train/test
        return split_dataset['train'], split_dataset['test']
    

    def setup_model_for_training(self, num_labels: int = 3, source_model_path: str = None):
        
        # 1. Грузим базовую модель (скелет) в 8-битном режиме (экономия памяти GPU)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=num_labels,
            load_in_8bit=True,                           # Включаем квантование (требует bitsandbytes)
            device_map="auto"                            # Авто-распределение по GPU
        )
        
        if source_model_path and os.path.exists(source_model_path):
            # 2a. Дообучение: Грузим существующий адаптер и включаем режим тренировки
            print(f"Загрузка существующих адаптеров из {source_model_path}...")
            self.model = PeftModel.from_pretrained(base_model, source_model_path, is_trainable=True)
        else:
            # 2b. С нуля: Создаем новую конфигурацию LoRA
            print("Инициализация новых LoRA адаптеров...")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,              # Тип задачи: Классификация последовательностей
                r=32,                                    # Ранг матриц адаптеров (больше = точнее, но тяжелее)
                lora_alpha=64,                           # Коэффициент масштабирования
                lora_dropout=0.1,                        # Регуляризация
                target_modules=["query", "value"]        # Куда внедряем адаптеры (слои внимания)
            )
            self.model = get_peft_model(base_model, lora_config) # Оборачиваем базу в LoRA
        
        self.model.print_trainable_parameters()          # Выводим, сколько % параметров реально обучается
        return self.model
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        output_dir: str = "./trained_models_temp", 
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-4
    ) -> Dict:
        
        # Конфигурация гиперпараметров обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,                           # L2 регуляризация
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,                            # Как часто писать логи
            eval_strategy="epoch",                       # Проверка качества в конце каждой эпохи
            save_strategy="epoch",                       # Сохранение чекпоинта в конце каждой эпохи
            save_total_limit=1,                          # Храним только 1 последний чекпоинт (экономим место)
            load_best_model_at_end=True,                 # В конце загружаем лучшую версию модели
            metric_for_best_model="eval_loss",           # Лучшая = где меньше loss
            greater_is_better=False,
            report_to="none"                             # Отключаем WandB/MLflow (для локального запуска)
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer) # Сборщик батчей с выравниванием

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        train_result = trainer.train()                   # ЗАПУСК ОБУЧЕНИЯ
        
        # Формируем отчет об обучении
        training_info = {
            "timestamp": datetime.now().isoformat(),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_loss": train_result.training_loss,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset)
        }
        self.training_history.append(training_info)
        
        # Удаляем временные чекпоинты (checkpoint-500...), чтобы не забить диск
        shutil.rmtree(f"{output_dir}/checkpoint-*", ignore_errors=True)
        
        return training_info
    
 
    def save_model(self, custom_name: str, base_path: str = "./trained_models"):
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        full_path = os.path.join(base_path, custom_name)
        os.makedirs(full_path, exist_ok=True)
        
        self.model.save_pretrained(full_path)            # Сохраняем веса адаптера (LoRA)
        self.tokenizer.save_pretrained(full_path)        # Сохраняем конфиг токенизатора
        
        # Сохраняем историю обучения рядом с моделью
        history_path = os.path.join(full_path, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return full_path
    
    def load_trained_model(self, model_dir: str):        # Метод загрузки для проверки/инференса
        from peft import PeftModel
        
        # Загружаем базу (скелет) в режиме инференса (is_trainable=False по дефолту)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=3,
            load_in_8bit=True,
            device_map="auto"
        )
        
        # Навешиваем обученный адаптер
        self.model = PeftModel.from_pretrained(base_model, model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.eval()                                # Переключаем в режим оценки (выключаем dropout)
        
        # Подгружаем историю, если есть
        history_path = os.path.join(model_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        return self.model
