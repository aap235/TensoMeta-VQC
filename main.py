import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Model import TextClassifier
from transformers import BertTokenizer, DataCollatorWithPadding
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # важно для сервера
import matplotlib.pyplot as plt

def plot_metrics_per_epoch(log_dir):
    # Найти последнюю версию
    versions = [d for d in os.listdir(log_dir) if d.startswith("version_")]
    if not versions:
        print("No logs found!")
        return
    latest_version = sorted(versions, key=lambda x: int(x.split('_')[-1]))[-1]
    csv_path = os.path.join(log_dir, latest_version, "metrics.csv")
    
    df = pd.read_csv(csv_path)

    # Удаляем строки без эпохи (иногда бывают)
    df = df.dropna(subset=['epoch'])
    df['epoch'] = df['epoch'].astype(int)

    # Группируем по эпохе: берём последнее значение в эпохе (или среднее)
    # В Lightning при log_every_n_steps > 1 может быть несколько строк на эпоху.
    # Мы возьмём последнюю запись в эпохе — она соответствует концу эпохи.
    df_epoch = df.groupby('epoch').last().reset_index()

    # Построение
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    epochs = df_epoch['epoch']

    # Loss
    if 'train_loss' in df_epoch.columns:
        ax[0].plot(epochs, df_epoch['train_loss'], marker='o', label='Train Loss')
    if 'val_loss' in df_epoch.columns:
        ax[0].plot(epochs, df_epoch['val_loss'], marker='o', label='Val Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss per Epoch')
    ax[0].legend()
    ax[0].grid(True)

    # Accuracy
    if 'train_acc' in df_epoch.columns:
        ax[1].plot(epochs, df_epoch['train_acc'], marker='o', label='Train Acc')
    if 'val_acc' in df_epoch.columns:
        ax[1].plot(epochs, df_epoch['val_acc'], marker='o', label='Val Acc')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy per Epoch')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(log_dir, latest_version, "metrics_per_epoch.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Metrics per epoch saved to: {plot_path}")

def main():
    #Загрузка датасета
    dataset = load_dataset("dbpedia_14")

    #Токенизация 
    tokenizer =  BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=512)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

    # Применяем токенизацию ко всему датасету
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]  # удаляем исходный текст
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    #Создаем DataLoader для обучения и валидации
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=64, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(tokenized_datasets["test"], batch_size=64, collate_fn=data_collator)

    #Обучение
    model = TextClassifier(num_classes=14, lr=1e-4)
    logger = pl.loggers.CSVLogger("logs", name="text_classification")
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=2,
        log_every_n_steps=50,
        logger=logger,
    )
    trainer.fit(model, train_loader, val_loader)
    plot_metrics_per_epoch("logs/text_classification")
    
if __name__ == "__main__":
    main()