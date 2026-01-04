import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from config import settings

class SentimentModel:
    def __init__(self):
        print("🚀 Инициализация модели...")
        
        # 1. Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)

        # 2. Конфиг для экономии памяти (8-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        # 3. Загружаем БАЗОВУЮ модель (скелет)
        print(f"📦 Loading Base Model: {settings.base_model_name}...")
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            settings.base_model_name,
            num_labels=3,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # 4. Инициализируем PEFT с ДЕФОЛТНЫМ адаптером
        # Мы даем ему имя "default", чтобы потом легко к нему возвращаться
        print(f"🔌 Loading Default Adapter: {settings.adapter_name}...")
        try:
            self.model = PeftModel.from_pretrained(
                self.base_model,
                settings.adapter_name,
                adapter_name="default", 
                is_trainable=False
            )
            print("✅ Default adapter loaded and active.")
        except Exception as e:
            print(f"❌ Error loading default adapter: {e}")
            # Если не вышло, модель остается просто оберткой над базой
            self.model = self.base_model

        self.active_adapter_name = "default"

    def switch_model(self, model_name: str):
        """
        Переключает активный адаптер, используя set_adapter.
        Не перезагружает базовую модель.
        """
        # 1. Определяем целевое имя адаптера
        # Если пришло None, "Default" или пустая строка -> используем "default"
        target_adapter = "default"
        if model_name and model_name not in ["Default", "Base", "default"]:
            target_adapter = model_name

        # 2. Если мы уже на этом адаптере - выходим
        if self.active_adapter_name == target_adapter:
            return

        print(f"🔄 Switching adapter to: '{target_adapter}'...")

        # 3. Если хотим вернуться к дефолтному
        if target_adapter == "default":
            try:
                self.model.set_adapter("default")
                self.active_adapter_name = "default"
                print("✅ Switched to Default adapter.")
            except Exception as e:
                print(f"❌ Failed to switch to default: {e}")
            return

        # 4. Если это кастомная модель
        # Сначала проверяем, загружена ли она уже в память
        if target_adapter in self.model.peft_config:
            self.model.set_adapter(target_adapter)
            self.active_adapter_name = target_adapter
            print(f"✅ Switched to cached adapter: {target_adapter}")
        else:
            # Если в памяти нет, пробуем загрузить с диска
            adapter_path = f"./trained_models/{target_adapter}"
            if not os.path.exists(adapter_path):
                print(f"⚠️ Путь адаптера не найден: {adapter_path}. Остается на текущем.")
                return

            try:
                print(f"📂 Загрузка новой модели из диска: {target_adapter}")
                self.model.load_adapter(adapter_path, adapter_name=target_adapter)
                self.model.set_adapter(target_adapter)
                self.active_adapter_name = target_adapter
                print(f"✅ Загружена и переключена на: {target_adapter}")
            except Exception as e:
                print(f"❌ Ошибка загрузки адаптера {target_adapter}: {e}")
                # Если ошибка, пытаемся вернуться на дефолт
                self.model.set_adapter("default")
                self.active_adapter_name = "default"

    def predict(self, text: str, model_name: str = None) -> dict:
        # Сначала переключаем адаптер
        self.switch_model(model_name)

        if not text:
            return {"label": "neutral", "score": 0.0}

        # Токенизация
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        
        # Перенос на GPU
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        score, label_id = torch.max(probs, dim=1)

        return {
            "label": f"LABEL_{label_id.item()}",
            "score": float(score.item())
        }

# Singleton
sentiment_model = None

def get_sentiment_model() -> SentimentModel:
    global sentiment_model
    if sentiment_model is None:
        sentiment_model = SentimentModel()
    return sentiment_model