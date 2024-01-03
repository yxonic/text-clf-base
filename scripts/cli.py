from lightning.pytorch.cli import LightningCLI

from text_clf_base.data import TextClfData
from text_clf_base.model import TextClf

if __name__ == "__main__":
    LightningCLI(TextClf, TextClfData)
