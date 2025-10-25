import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from MetaVQC import TensorMeta_VQC
import torch
from torchmetrics import Accuracy

class TextClassifier(pl.LightningModule):
    def __init__(self, n_wires = 4, n_layers = 10, num_class=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TensorMeta_VQC(n_wires = n_wires, n_layers = n_layers, num_class=num_class)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_class)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_class)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])
        self.train_acc(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])
        self.val_acc(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)