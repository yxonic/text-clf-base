# load dataset with pytorch dataloader
import polars as pl
import tokenizers
import torch
from torch.utils.data import Dataset

from text_clf_base.util import Lines


class TextClfDataset(Dataset):
    def __init__(self, data_path, label_path, max_length=128):
        self.data = Lines(data_path)
        self.label = pl.read_csv(label_path, has_header=False)["column_1"]
        self.tokenizer = tokenizers.Tokenizer.from_pretrained("bert-base-chinese")
        self.tokenizer.enable_padding(length=max_length)
        self.tokenizer.enable_truncation(max_length=max_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.tokenizer.encode("[CLS] " + self.data[idx]).ids),
            torch.tensor(self.label[idx]),
        )
