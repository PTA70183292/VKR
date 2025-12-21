import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd
from typing import List, Dict, Tuple
from config import settings
import json
from datetime import datetime

class SentimentTrainer:
    def __init__(self, base_model_name: str = None):
        self.base_model_name = base_model_name or settings.base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = None
        self.training_history = []
    
    def load_dataset_from_csv(self, file_path: str) -> Dataset:

        df = pd.read_csv(file_path)
        

        required_columns = ['text', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV должен содержать колонки: {required_columns}")
        

        if df['label'].dtype == 'object':
            label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
            df['label'] = df['label'].map(label_mapping)
        
        dataset = Dataset.from_pandas(df[['text', 'label']])
        return dataset
    
    def load_dataset_from_dict(self, data: List[Dict[str, any]]) -> Dataset:

        df = pd.DataFrame(data)
        return Dataset.from_pandas(df)
    
    def preprocess_function(self, examples):

        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=128
        )
    
    def prepare_dataset(self, dataset: Dataset, test_size: float = 0.2) -> Tuple[Dataset, Dataset]:

        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['text']
        )
        

        split_dataset = tokenized_dataset.train_test_split(test_size=test_size)
        return split_dataset['train'], split_dataset['test']
    
    def setup_model_for_training(self, num_labels: int = 3):

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=num_labels,
            load_in_8bit=True,
            device_map="auto"
        )
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        output_dir: str = "./trained_model",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
        save_steps: int = 500
    ) -> Dict:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none"
        )
        

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        train_result = trainer.train()
        
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
        
        return training_info
    
    def evaluate(self, eval_dataset: Dataset) -> Dict:
        training_args = TrainingArguments(
            output_dir="./tmp",
            per_device_eval_batch_size=8,
            report_to="none"
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        eval_results = trainer.evaluate()
        return eval_results
    
    def save_model(self, output_dir: str = "./saved_model"):
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return output_dir
    
    def load_trained_model(self, model_dir: str):

        from peft import PeftModel
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=3,
            load_in_8bit=True,
            device_map="auto"
        )
        
        self.model = PeftModel.from_pretrained(base_model, model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        history_path = os.path.join(model_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        return self.model
    
    def predict_text(self, text: str) -> Dict:

        if self.model is None:
            raise ValueError("Модель не загружена")
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            pred_score = probs[0][pred_label].item()
        
        return {
            "label": f"LABEL_{pred_label}",
            "score": float(pred_score)
        }
