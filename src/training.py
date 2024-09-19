import torch
import numpy as np
import lightning as L
from lightning.pytorch import loggers as pl_loggers

###### LIGHTNING MODULE
# Integrate lightning
class LightningModel(L.LightningModule):
    def __init__(
        self, 
        model, 
        loss_fn_class, 
        optimizer_class, 
        lr
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.loss_fn = loss_fn_class(reduction='sum')
        self.optimizer_class = optimizer_class
        self.lr = lr

    def forward(self, batch):
        preds = self.model(batch)
        return preds

    def print_for_debug(
        self, 
        batch, 
        preds, 
        n_preds,
        masked_preds, 
        targets, 
        masked_targets,
        total_loss, 
        avg_loss,
        correct,
        n_correct,
        acc
    ):
        for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    print(f"{key} : shape {val.shape}")
        print(f"n_preds : {n_preds}")
        print(f"preds : shape {preds.shape}; mean & std ({preds.mean():.3f}, {preds.std():.3f})")
        print(f"masked_preds : {masked_preds.shape}")
        print(f"targets : {targets.shape}; mean & std ({targets.mean():.3f}, {targets.std():.3f})")
        print(f"masked_targets : {masked_targets.shape}")
        print(f"total_loss : {total_loss}")
        print(f"avg_loss : {avg_loss}")
        print(f"correct : {correct.shape}")
        print(f"n_correct : {n_correct}")
        print(f"acc : {acc}")

    def training_step(self, batch, batch_idx=0, debug=False):
        preds = self(batch)
        targets = batch['targets']
        padding_masks = batch['padding_masks']
        
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
            self.print_for_debug(
                batch, 
                preds, 
                n_preds,
                masked_preds, 
                targets, 
                masked_targets,
                total_loss, 
                avg_loss,
                correct,
                n_correct,
                acc
            )
        # Log
        self.log("train_loss", avg_loss.item(), on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return avg_loss    
    
    def validation_step(self, batch, batch_idx=0, debug=False):
        preds = self(batch)
        targets = batch['targets']
        padding_masks = batch['padding_masks']
        
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
        
        if debug:
            self.print_for_debug(
                batch, 
                preds, 
                n_preds,
                masked_preds, 
                targets, 
                masked_targets,
                total_loss, 
                avg_loss,
                correct,
                n_correct,
                acc
            )
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
