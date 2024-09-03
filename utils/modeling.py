print(" * Importing ... ")
import sqlite3
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from torchsummary import summary
import lightning as L
from lightning.pytorch import loggers as pl_loggers


# Turns given sequence of features into one-hot encoded sequence of features
def make_ohe(sequence : torch.Tensor, ohe_size=32, stack=True) -> torch.Tensor:
    # NOTE: NumPy has better support for mapping
    n, T, d = sequence.shape
    ohe_sequence = []

    for i in range(d):
        # (1) Give each feature value an ID
        feature_sequence = torch.Tensor.numpy(sequence[:, :, i])
        unique_values = np.unique(feature_sequence)
        id_mapping = dict(zip(unique_values, range(len(unique_values))))
        vectorized_id_mapping = np.vectorize(id_mapping.get)
        
        # (2) Map features to ID's; get OHE according to ID's
        feature_id_sequence = vectorized_id_mapping(feature_sequence)
        feature_ohe_sequence = np.eye(ohe_size)[feature_id_sequence]
        ohe_sequence.append(torch.from_numpy(feature_ohe_sequence).to(torch.float))
    
    if stack:
        return torch.cat(ohe_sequence, dim=-1)
    return ohe_sequence


##### MODEL ARCHITECTURE (PyTorch)
class TransformerLayerB(nn.Module):
    def __init__(self, 
                 input_dim_kq,  
                 embed_dim_kq,
                 input_dim_v,
                 embed_dim_v,
                 n_heads=1):
        super(TransformerLayerB, self).__init__()

        self.input_dim_kq = input_dim_kq
        self.embed_dim_kq = embed_dim_kq
        self.input_dim_v = input_dim_v
        self.embed_dim_v = embed_dim_v

        self.W_k = nn.Linear(input_dim_kq, embed_dim_kq)
        self.W_q = nn.Linear(input_dim_kq, embed_dim_kq)
        self.W_v = nn.Linear(input_dim_v, embed_dim_v)
        self.n_heads = n_heads
        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dim_v, 4*embed_dim_v),
            nn.GELU(),
            nn.Linear(4*embed_dim_v, embed_dim_v)
        )

    def forward(self, kq, v) -> torch.Tensor:
        # FOR TIME BEING:
        #  * keys are OHE matchups
        #  * values are stats
        n, T, _ = kq.shape
        n_heads = self.n_heads

        keys = self.W_k(kq)
        queries = self.W_q(kq)
        values = self.W_v(v)

        keys = keys.view(n, T, n_heads, -1).transpose(1, 2).contiguous()
        queries = queries.view(n, T, n_heads, -1).transpose(1, 2).contiguous()
        values = values.view(n, T, n_heads, -1).transpose(1, 2).contiguous()

        attn_weights = queries @ keys.transpose(-2, -1)
        
        # Causal mask, cannot attend to diagonal
        # Test; allowing the first game to attend to itself
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=0)
        mask[0,0] = 0 # To get rid of NaN's
        attn_scores = F.softmax(attn_weights + mask, dim=-1)
        
        # v' = A @ v
        values = attn_scores @ values

        values = values.transpose(2, 1).contiguous().view(n, T, -1)
        values = self.ff_layer(values)
        
        return values


class Model(nn.Module):
    def __init__(self, 
                 input_dim_kq,  
                 embed_dim_kq,
                 input_dim_v,
                 embed_dim_v,
                 output_dim,
                 n_heads=1):
        super(Model, self).__init__()

        self.input_dim_kq = input_dim_kq
        self.embed_dim_kq = embed_dim_kq
        self.input_dim_v = input_dim_v
        self.embed_dim_v = embed_dim_v
        self.output_dim = output_dim

        self.n_heads = n_heads
        self.transformer_layer = TransformerLayerB(
            input_dim_kq, 
            embed_dim_kq,
            input_dim_v,
            embed_dim_v,
            n_heads)
        self.output_layer = nn.Linear(embed_dim_v, output_dim)

    def forward(self, kq, v):
        v = self.transformer_layer(kq, v)
        return self.output_layer(v)

    def predict(self, kq, v):
        return (self.forward(kq, v) > 0).to(torch.int)


###### LIGHTNING MODULE
# Integrate lightning
class LightningModel(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer_class, lr):
        super().__init__()
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.lr = lr

    def forward(self, kq, v):
        kq = make_ohe(kq)
        return self.model(kq, v)

    def training_step(self, batch, batch_idx):
        kq, v, targets = batch
        preds = self(kq, v)
        loss = self.loss_fn(preds, targets)
        loss = self.loss_fn(preds, targets)
        acc = (torch.sign(preds) == torch.sign(targets)).to(torch.float).mean().item()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss    
    
    def validation_step(self, batch, batch_idx):
        kq, v, targets = batch
        preds = self(kq, v)
        loss = self.loss_fn(preds, targets)
        acc = (torch.sign(preds) == torch.sign(targets)).to(torch.float).mean().item()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), 
            lr=self.lr
        )
        return {"optimizer" : optimizer}
