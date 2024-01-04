from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from text_clf_base.data import TextClfData
from text_clf_base.model import TextClf


def main():
    # some default setup
    checkpoint = ModelCheckpoint(
        filename="{epoch:03d}-{f1:.3f}-{cross_entropy:.3f}",
        monitor="cross_entropy",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="cross_entropy", mode="min", patience=10)

    trainer_defaults = {
        "callbacks": [checkpoint, early_stopping],
        "max_epochs": 500,
    }

    # run cli
    LightningCLI(TextClf, TextClfData, trainer_defaults=trainer_defaults)


if __name__ == "__main__":
    main()
