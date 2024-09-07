import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import lightning as L
from lightning.pytorch import loggers as pl_loggers

# Turns given sequence of features into one-hot encoded sequence of features
def make_ohe(
    sequence : torch.Tensor, 
    shuffle : bool,
    ohe_size=32, 
    stack=True, 
) -> torch.Tensor:
    # NOTE: NumPy has better support for mapping
    n, T, d = sequence.shape
    ohe_sequence = []

    for i in range(d):
        # (1) Give each feature value an ID
        feature_sequence = torch.Tensor.numpy(sequence[:, :, i])
        unique_values = np.unique(feature_sequence)
        if shuffle:
            np.random.shuffle(unique_values)
        id_mapping = dict(zip(unique_values, range(len(unique_values))))
        vectorized_id_mapping = np.vectorize(id_mapping.get)
        
        # (2) Map features to ID's; get OHE according to ID's
        feature_id_sequence = vectorized_id_mapping(feature_sequence)
        feature_ohe_sequence = np.eye(ohe_size)[feature_id_sequence]
        ohe_sequence.append(torch.from_numpy(feature_ohe_sequence).to(torch.float))
    
    if stack:
        return torch.cat(ohe_sequence, dim=-1)
    return ohe_sequence

def make_padding_masks(batch):
    make_padding_mask = lambda sequence : torch.ones(sequence.shape[0])
    padding_masks = list(map(make_padding_mask, batch))
    padding_masks = pad_sequence(padding_masks, batch_first=True, padding_value=0)
    return padding_masks.unsqueeze(dim=-1)

def normalize(values : torch.Tensor) -> torch.Tensor:
    n, T, d = values.shape
    # Want to normalize across 'T' dimension
    means = values.mean(dim=1, keepdim=True)
    stds = values.std(dim=1, keepdim=True)    
    normalized_values = (values - means) / stds
    return normalized_values, means, stds

def denormalize(
    normalized_values : torch.Tensor, 
    means : torch.Tensor, 
    stds : torch.Tensor
):
    values = (normalized_values * stds) + means
    return values

###### LIGHTNING MODULE
# Integrate lightning
class LightningModel(L.LightningModule):
    def __init__(
        self, 
        model, 
        loss_fn_class, 
        optimizer_class, 
        lr, 
        normalize=False,
        debug=False
    ):
        super().__init__()
        
        self.model = model
        self.loss_fn = loss_fn_class(reduction='sum')
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.normalize=normalize
        self.debug = debug

    def forward(self, kq, v):
        if self.normalize:
            v, means, stds = normalize(v)
            preds = self.model(kq, v)
            preds = denormalize(preds, means, stds)
        else:
            preds = self.model(kq, v)
        return preds

    def training_step(self, batch, batch_idx):
        debug = self.debug
        
        kq, v, targets, padding_masks = batch
        preds = self(kq, v)
        
        # To make calculations wrt padding
        n_preds = padding_masks.sum().item()
        
        # Mask out irrelevant, calculate loss (wrt padding)
        masked_preds = preds * padding_masks
        masked_targets = targets * padding_masks
        total_loss = self.loss_fn(masked_preds, masked_targets)
        avg_loss = total_loss / n_preds
        
        # Calculate accuracy (wrt padding)
        correct = (torch.sign(preds) == torch.sign(targets)).to(torch.float) * padding_masks
        n_correct = correct.sum().item()
        acc = n_correct / n_preds

        if debug:
            print(f"kq : {kq.shape}")
            print(f"v : {v.shape}")
            print(f"targets : {targets.shape}, {targets.mean()}, {targets.std()}")
            print(f"masks : {padding_masks.shape}")
            print(f"preds : {preds.shape}, {preds.mean()}, {preds.std()}")
            print(f"n_preds : {n_preds}")
            print(f"masked_preds : {masked_preds.shape}")
            print(f"masked_targets : {masked_targets.shape}")
            print(f"total_loss : {total_loss}")
            print(f"avg_loss : {avg_loss}")
            print(f"correct : {correct.shape}")
            print(f"n_correct : {n_correct}")
            print(f"acc : {acc}")
            input()

        # Log
        self.log("train_loss", avg_loss.item(), on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return avg_loss    
    
    def validation_step(self, batch, batch_idx):
        kq, v, targets, padding_masks = batch
        preds = self(kq, v)
        
        # Deal with padding
        n_preds = padding_masks.sum().item()
        
        # Mask out irrelevant, calculate loss (wrt padding)
        masked_preds = preds * padding_masks
        masked_targets = targets * padding_masks
        total_loss = self.loss_fn(masked_preds, masked_targets)
        avg_loss = total_loss / n_preds
        
        # Calculate loss (wrt padding)
        total_loss = self.loss_fn(masked_preds, masked_targets)
        avg_loss = total_loss / n_preds
        
        # Calculate accuracy (wrt padding)
        correct = (torch.sign(preds) == torch.sign(targets)).to(torch.float) * padding_masks
        n_correct = correct.sum().item()
        acc = n_correct / n_preds
        
        # Log
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), 
            lr=self.lr
        )
        return {"optimizer" : optimizer}
