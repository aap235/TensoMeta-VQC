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
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def plot_metrics_per_epoch(log_dir):
    # Настройка стиля под научную публикацию
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "serif",          # serif как в статьях
        "text.usetex": False,            # если нет LaTeX — False; если есть — можно True
        "figure.figsize": (8, 3.5),      # компактный, как в колонке статьи
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    # Найти последнюю версию
    versions = [d for d in os.listdir(log_dir) if d.startswith("version_")]
    if not versions:
        print("No logs found!")
        return
    latest_version = sorted(versions, key=lambda x: int(x.split('_')[-1]))[-1]
    csv_path = os.path.join(log_dir, latest_version, "metrics.csv")
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['epoch'])
    df['epoch'] = df['epoch'].astype(int)
    df_epoch = df.groupby('epoch').last().reset_index()

    epochs = df_epoch['epoch']

    # Цвета и стили, различимые в Ч/Б
    train_style = {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'markersize': 4}
    val_style = {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'markersize': 4}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # --- Loss ---
    if 'train_loss' in df_epoch.columns:
        ax1.plot(epochs, df_epoch['train_loss'], label='Train', **train_style)
    if 'val_loss' in df_epoch.columns:
        ax1.plot(epochs, df_epoch['val_loss'], label='Validation', **val_style)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Accuracy ---
    if 'train_acc' in df_epoch.columns:
        ax2.plot(epochs, df_epoch['train_acc'], label='Train', **train_style)
    if 'val_acc' in df_epoch.columns:
        ax2.plot(epochs, df_epoch['val_acc'], label='Validation', **val_style)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Общая легенда (можно и отдельно, но так компактнее)
    handles, labels = ax1.get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # место под легенду сверху

    plot_path = os.path.join(log_dir, latest_version, "metrics_per_epoch.pdf")  # PDF — лучше для статей
    plt.savefig(plot_path)
    
    # Также сохраняем PNG для быстрого просмотра
    plt.savefig(plot_path.replace(".pdf", ".png"), dpi=300)
    
    plt.close()
    print(f"plot saved to: {plot_path}")

def main():
    #Загрузка датасета
    dataset = load_dataset("dbpedia_14")

    #Токенизация 
    tokenizer =  BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=512)
    def tokenize_function(examples):
        return tokenizer(
            examples["content"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

    # Применяем токенизацию ко всему датасету
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["content"]  # удаляем исходный текст
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    #Создаем DataLoader для обучения и валидации
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=64, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(tokenized_datasets["test"], batch_size=64, collate_fn=data_collator)

    #Обучение
    model = TextClassifier(n_wires = 10, n_layers= 2, num_class=14, lr=1e-3)
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