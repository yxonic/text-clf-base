# load dataset with pytorch dataloader
import os

import lightning as L
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from text_clf_base.util import Lines, get_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextClfDataset(Dataset):
    def __init__(self, data_path, label_path, tokenizer):
        self.data = Lines(data_path)
        self.label = pl.read_csv(label_path, has_header=False)["column_1"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.tokenizer.encode(self.data[idx]).ids),
            torch.tensor(self.label[idx]),
        )


class TextClfData(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=32, max_length=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = get_tokenizer()
        self.tokenizer.enable_padding(length=self.max_length)
        self.tokenizer.enable_truncation(max_length=self.max_length)

    def setup(self, stage):  # noqa: ARG002
        self.train_dataset = TextClfDataset(
            os.path.join(self.data_dir, "train_text.txt"),
            os.path.join(self.data_dir, "train_label.txt"),
            self.tokenizer,
        )
        self.val_dataset = TextClfDataset(
            os.path.join(self.data_dir, "val_text.txt"),
            os.path.join(self.data_dir, "val_label.txt"),
            self.tokenizer,
        )
        self.test_dataset = TextClfDataset(
            os.path.join(self.data_dir, "test_text.txt"),
            os.path.join(self.data_dir, "test_label.txt"),
            self.tokenizer,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2, persistent_workers=True)
