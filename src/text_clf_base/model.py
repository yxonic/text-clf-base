import lightning as L
from torch import nn, optim


class TextClf(L.LightningModule):
    def __init__(self, vocab_size, d_model=512, num_layers=4, nhead=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation="gelu", batch_first=True),
            num_layers=num_layers,
        )
        self.output = nn.Linear(d_model, 1)

    def training_step(self, batch, _):
        x, y = batch
        y_ = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_, y.float())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # following RoBERTa
        return optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-6)

    def forward(self, x):
        x = self.embed(x)
        h = self.model(x)[:, 0, :]
        return self.output(h).squeeze(-1)
