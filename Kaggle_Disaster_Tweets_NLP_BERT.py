import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import evaluate
from datasets import Dataset, load_dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    pipeline,
)

def preprocess_data(train: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    """ 
    Загружает данные, удаляет дубликаты в текстах, 
    делит на обучающую и валидационную выборки, переводит в Datasets.
    """

    train = train[['text', 'target']]
    train.columns = ['text', 'label']
    train = train[~train['text'].duplicated(keep=False)]

    train, val = train_test_split(train, test_size=0.2, random_state=42)
    train = Dataset.from_pandas(train)
    val = Dataset.from_pandas(val)

    return train, val

def tokenize_datasets(
        tokenizer: PreTrainedTokenizerBase,
        train: Dataset, 
        val: Dataset
        ) -> tuple[Dataset, Dataset]:
    """Токкенизирует тексты."""

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)
    
    train = train.map(tokenize, batched=True)
    val = val.map(tokenize, batched=True)

    return train, val

def train_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train: Dataset, 
        val: Dataset,
        metric_name: str,
        batch_size: int,
        model_save_name: str
        ) -> Trainer:
    """Тренирует модель для задачи классификации текста и возвращает тренер."""

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        save_total_limit=1,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        weight_decay=0.05
    )

    metric = evaluate.load(metric_name)
    def compute_metrics(eval_pred):
        """Высчитывает метрику."""

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)

        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    os.makedirs('models', exist_ok=True)
    trainer.save_model(os.path.join('models', model_save_name))

    return trainer

def make_prediction(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        test: pd.DataFrame
        ) -> pd.DataFrame:
    """Предсказание и сохранение результата в DataFrame."""

    texts = test['text'].to_list()

    pred_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        padding=True
    )

    preds = pred_pipe(texts, batch_size=16)
    labels = [1 if p['label'] == 'LABEL_1' else 0 for p in preds]
    result = pd.DataFrame(
        {'id': test['id'],
        'target': labels}
    )

    return result


def final_pipeline(
        train: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        metric: str,
        test: pd.DataFrame,
        batch_size: int,
        model_save_name: str
):

    # Выводит информацию об используемом устройстве (CPU или GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
        
    train, val = preprocess_data(train)
    train, val = tokenize_datasets(tokenizer, train, val)
    trainer = train_model(model, tokenizer, train, val, metric, batch_size, model_save_name)

    predictions = make_prediction(trainer.model, tokenizer, test)

    return predictions


train = pd.read_csv('datasets/train.csv')    
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
metric_name = 'f1'
test = pd.read_csv('datasets/test.csv')
batch_size = 16
model_save_name = 'disaster_tweets_BERT_model'

result = final_pipeline(
    train, 
    tokenizer, 
    model, 
    metric_name, 
    test,
    batch_size, 
    model_save_name
    )