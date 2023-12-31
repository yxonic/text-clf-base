import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics as M
from torch import nn, optim

from text_clf_base.util import get_tokenizer


class TextClf(L.LightningModule):
    def __init__(self, d_model=512, num_layers=4, nhead=8):
        super().__init__()

        # lightning setup
        self.save_hyperparameters()
        self.example_input_array = torch.ones(32, 128).long()

        # store evaluation results
        self.predictions = []

        # model
        vocab_size = get_tokenizer().get_vocab_size()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation="gelu", batch_first=True),
            num_layers=num_layers,
        )
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        h = self.model(x)[:, 0, :]
        return self.output(h).squeeze(-1)

    def configure_optimizers(self):
        # following RoBERTa
        return optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-6)

    def training_step(self, batch, _):
        x, y = batch
        y_ = self(x)
        loss = F.binary_cross_entropy_with_logits(y_, y.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_ = self(x)
        cross_entropy = F.binary_cross_entropy_with_logits(y_, y.float())
        self.log("cross_entropy", cross_entropy, sync_dist=True)
        self.predictions.append(torch.sigmoid(y_))
        return y_

    def test_step(self, batch):
        self.validation_step(batch)

    def calc_epoch_metrics(self, dataloader):
        y_pred = torch.cat(self.predictions)
        self.predictions.clear()
        y_true = torch.tensor(dataloader.dataset.label[: y_pred.size(0)])
        y_pred = y_pred.to(y_true.device)
        self.log("precision", M.functional.precision(y_pred, y_true, "binary"), sync_dist=True)
        self.log("recall", M.functional.recall(y_pred, y_true, "binary"), sync_dist=True)
        self.log("f1", M.functional.f1_score(y_pred, y_true, "binary"), sync_dist=True)
        self.log("auc", M.functional.auroc(y_pred, y_true, "binary"), sync_dist=True)

    def on_validation_epoch_end(self):
        self.calc_epoch_metrics(self.trainer.val_dataloaders)

    def on_test_epoch_end(self):
        self.calc_epoch_metrics(self.trainer.test_dataloaders)
