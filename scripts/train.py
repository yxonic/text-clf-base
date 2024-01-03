import os

import lightning as L
from torch import utils

from text_clf_base.data import TextClfDataset
from text_clf_base.model import TextClf

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET = "sample"


def train():
    # load datasets
    train_dataset = TextClfDataset(f"data/{DATASET}/train_text.txt", f"data/{DATASET}/train_label.txt")
    train_loader = utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True
    )
    val_dataset = TextClfDataset(f"data/{DATASET}/valid_text.txt", f"data/{DATASET}/valid_label.txt")
    val_loader = utils.data.DataLoader(val_dataset, batch_size=32, num_workers=2, persistent_workers=True)
    test_dataset = TextClfDataset(f"data/{DATASET}/test_text.txt", f"data/{DATASET}/test_label.txt")
    test_loader = utils.data.DataLoader(test_dataset, batch_size=64, num_workers=2, persistent_workers=True)

    # build model
    model = TextClf(train_dataset.tokenizer.get_vocab_size(), d_model=128, num_layers=4, nhead=8)

    # train
    model.train()
    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # test
    trainer.test(dataloaders=test_loader, ckpt_path="last")


if __name__ == "__main__":
    train()
