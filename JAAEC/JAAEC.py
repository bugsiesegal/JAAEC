"""Main module."""
import math

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.optim import Adam


class AmazingEncoder(pl.LightningModule):
    def __init__(self, input_shape, embedding_shape, num_layers=3, num_heads=8):
        super(AmazingEncoder, self).__init__()

        self.input_shape = input_shape
        self.embedding_shape = embedding_shape

        self.linear_expander = nn.Linear(input_shape[2], embedding_shape[1])

        self.linear_encoder = nn.Linear(input_shape[1], 1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_shape[1],
                                                   nhead=num_heads, dim_feedforward=input_shape[1],
                                                   batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor):
        x = self.linear_expander(x)

        x = self.encoder(x)

        x = x.reshape((x.shape[0], x.shape[2], x.shape[1]))

        x = self.linear_encoder(x)

        x = x.squeeze(2)

        return x


class AmazingDecoder(pl.LightningModule):
    def __init__(self, input_shape, embedding_shape, num_layers=3, num_heads=8):
        super(AmazingDecoder, self).__init__()

        self.input_shape = input_shape
        self.embedding_shape = embedding_shape

        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_shape[1],
                                                   nhead=num_heads,
                                                   dim_feedforward=input_shape[1],
                                                   batch_first=True)

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.linear_decoder = nn.Linear(1, input_shape[1])

        self.linear_expander = nn.Linear(embedding_shape[1], input_shape[2])

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(2)

        x = self.linear_decoder(x)

        x = x.reshape((x.shape[0], x.shape[2], x.shape[1]))

        x = self.decoder(x, x)

        x = self.linear_expander(x)

        return x


class AmazingAutoEncoder(pl.LightningModule):
    def __init__(self, input_shape, embedding_shape, learning_rate, num_layers=3, num_heads=8):
        super(AmazingAutoEncoder, self).__init__()

        self.encoder = AmazingEncoder(input_shape, embedding_shape, num_layers, num_heads)

        self.decoder = AmazingDecoder(input_shape, embedding_shape, num_layers, num_heads)

        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)

        x = self.decoder(x)

        return x

    def training_step(self, batch, batch_idx):
        x = batch

        x_hat = self(x)

        loss = nn.MSELoss()(x_hat, x)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch

        x_hat = self(x)

        loss = nn.MSELoss()(x_hat, x)

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch

        x_hat = self(x)

        loss = nn.MSELoss()(x_hat, x)

        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr or self.learning_rate))
