import sys
import os
import importlib.util
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# External imports
import yaml
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader

#### STRING PROCESSING UTILITIES ####
def season_int_to_str(season : int) -> str:
    """Converts reference of season start to reference of entire season.

    The resulting string can be used to specify queries for the NBA API.
    Example use: season_to_str(2023) gives '2023-24'

    Args: 
        season: an int meant to reference the start of an NBA season 
    
    Returns:
        A string referencing the entire season, meant to be used to 
        query from the NBA API. Example use: season_to_str(2023) gives '2023-24'
    """    
    return f"{season}-{str(season + 1)[-2:]}"

def season_str_to_int(season_str : str) -> int:
    return int(season_str[:4])

def get_summary_from_main(main_col):
    stripped_col = main_col.replace('_for', '')
    stripped_col = stripped_col.replace('_ag', '')
    mean_col = stripped_col + '_mean'
    std_col = stripped_col + '_std'
    return mean_col, std_col

def get_main_to_summary_map(main_cols):
    map = {}
    for col in main_cols:
        mean_col, std_col = get_summary_from_main(col)
        map[col] = {
            'mean' : mean_col,
            'std' : std_col
        }
    return map

def rename_for_rolled_opp(col):
    if '_ag' in col:
        return col.replace('_ag', '_for_opp')
    elif '_for' in col:
        return col.replace('_for', '_ag_opp')
    return col + '_opp'

#### MODEL VISUALIZATION/EVALUATION ####    
def visualize_regression_performance(targets : np.ndarray, preds : np.ndarray):

    mse_per_instance = (targets - preds) ** 2

    # Plot histogram of MSE
    # Create a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 12))

    fig.suptitle('Visualization of Regression Performance', fontsize=16)

    # Next
    axs[0,0].hist(preds, bins=40, color='#ADD8E6', edgecolor='black', alpha=0.7)
    axs[0,0].set_title('Histogram of preds')
    axs[0,0].set_xlabel('preds')
    axs[0,0].set_ylabel('Frequency')
    axs[0,0].grid(axis='y', alpha=0.75)


    axs[0,1].hist(targets, bins=40, color='#0096FF', edgecolor='black', alpha=0.7)
    axs[0,1].set_title('Histogram of targets')
    axs[0,1].set_xlabel('targets')
    axs[0,1].set_ylabel('Frequency')
    axs[0,1].grid(axis='y', alpha=0.75)

    correct = np.sign(preds) == np.sign(targets)

    axs[1,0].scatter(targets[correct], preds[correct], sizes=[1], color='skyblue', label='Correct predictions')
    axs[1,0].scatter(targets[~correct], preds[~correct], sizes=[1], color='orange', label='Incorrect predictions')
    axs[1,0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], color='red', linestyle='--', label='y = y_pred')  # Reference line
    axs[1,0].set_title('Targets vs Predicted Values')
    axs[1,0].set_xlabel('Targets')
    axs[1,0].set_ylabel('Preds')
    axs[1,0].axhline(0, color='black', linewidth=1, linestyle='-')
    axs[1,0].axvline(0, color='black', linewidth=1, linestyle='-')
    axs[1,0].legend()
    axs[1,0].grid()
    axs[1,0].axis('equal')  # Equal scaling for better visualization

    axs[1,1].scatter(targets[correct], mse_per_instance[correct], color='skyblue', sizes=[1])
    axs[1,1].scatter(targets[~correct], mse_per_instance[~correct], color='orange', sizes=[1])
    axs[1,1].set_title('Targets vs MSE')
    axs[1,1].set_xlabel('Targets')
    axs[1,1].set_ylabel('Mean Squared Error (MSE)')
    axs[1,1].axvline(0, color='black', linewidth=1, linestyle='-')
    axs[1,1].axhline(0, color='black', linewidth=1, linestyle='-')
    axs[1,1].grid()

    plt.show()

def plot_training_metrics(
    mses_tr, 
    mses_val, 
    accs_tr, 
    accs_val,
    mse_range=(0.6, 1.0),
    mse_step=0.02,
    acc_range=(0.5, 0.7),
    acc_step=0.01
):
    
    best_mse_tr_idx = np.argmin(mses_tr)
    best_mse_val_idx = np.argmin(mses_val)
    best_acc_tr_idx = np.argmax(accs_tr)
    best_acc_val_idx = np.argmax(accs_val)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    fig.suptitle('TRAINING METRICS', fontsize=16)

    y_axis_mse = np.arange(mse_range[0], mse_range[1] + mse_step, mse_step)
    y_axis_acc = np.arange(acc_range[0], acc_range[1] + acc_step, acc_step)

    axes[0, 0].plot(mses_tr, color='y')
    axes[0, 0].set_title('train_loss')
    axes[0, 0].set_title('MSE TR vs EPOCH')
    axes[0, 0].set_xlabel('epoch')
    axes[0, 0].set_ylabel('mse_tr')
    axes[0, 0].set_yticks(y_axis_mse)
    axes[0, 0].set_ylim(min(y_axis_mse), max(y_axis_mse))
    axes[0, 0].grid(True)
    for y in np.arange(0.6, 1.01, 0.1):
        axes[0, 0].axhline(y, color='black', linestyle='-', linewidth=0.5) 

    axes[0, 0].plot(best_mse_val_idx, mses_tr[best_mse_val_idx], 'rx', markersize=8,
                    label=f"Best val epoch : mse_tr = {mses_tr[best_mse_val_idx]:.3f}")
    axes[0, 0].axhline(mses_tr[best_mse_val_idx], color='r', linestyle='--')
    axes[0, 0].axvline(best_mse_val_idx, color='r', linestyle='--')
    axes[0, 0].plot(best_mse_tr_idx, mses_tr[best_mse_tr_idx], color='r', marker='*',
                    markersize=8, label=f"Best mse_tr : {mses_tr[best_mse_tr_idx]:.3f}")
    axes[0, 0].legend()
    

    axes[0, 1].plot(mses_val, color='g')
    axes[0, 1].set_title('val_loss')
    axes[0, 1].set_title('MSE VAL vs EPOCH')
    axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylabel('mse_val')
    axes[0, 1].set_yticks(y_axis_mse)
    axes[0, 1].set_ylim(min(y_axis_mse), max(y_axis_mse))
    axes[0, 1].grid(True)
    for y in np.arange(0.6, 1.01, 0.1):
        axes[0, 1].axhline(y, color='black', linestyle='-', linewidth=0.5) 

    axes[0, 1].plot(best_mse_val_idx, mses_val[best_mse_val_idx], 'rx', markersize=8, 
                    label=f"Best val epoch : mse_val = {mses_val[best_mse_val_idx]:.3f}")
    axes[0, 1].axhline(mses_val[best_mse_val_idx], color='r', linestyle='--')
    axes[0, 1].axvline(best_mse_val_idx, color='r', linestyle='--')
    axes[0, 1].plot(best_mse_val_idx, mses_val[best_mse_val_idx], color='r', marker='*',
                    markersize=8, label=f"Best mse_val : {mses_val[best_mse_val_idx]:.3f}")
    axes[0, 1].legend()

    axes[1, 0].plot(accs_tr, color='b')
    axes[1, 0].set_title('train_acc')
    axes[1, 0].set_title('ACC TR vs EPOCH')
    axes[1, 0].set_xlabel('epoch')
    axes[1, 0].set_ylabel('acc_tr')
    axes[1, 0].set_yticks(y_axis_acc)
    axes[1, 0].set_ylim(min(y_axis_acc), max(y_axis_acc))
    axes[1, 0].grid(True)
    for y in np.arange(0.5, 0.71, 0.05):
        axes[1, 0].axhline(y, color='black', linestyle='-', linewidth=0.5) 

    axes[1, 0].plot(best_mse_val_idx, accs_tr[best_mse_val_idx], 'rx', markersize=8,
                    label=f"Best val epoch : acc_tr = {accs_tr[best_mse_val_idx]:.3f}")
    axes[1, 0].axhline(accs_tr[best_mse_val_idx], color='r', linestyle='--')
    axes[1, 0].axvline(best_mse_val_idx, color='r', linestyle='--')
    axes[1, 0].plot(best_acc_tr_idx, accs_tr[best_acc_tr_idx], color='r', marker='*',
                    markersize=8, label=f"Best acc_tr : {accs_tr[best_acc_tr_idx]:.3f}")
    axes[1, 0].legend()

    axes[1, 1].plot(accs_val, color='m')
    axes[1, 1].set_title('val_loss')
    axes[1, 1].set_title('ACC VAL vs EPOCH')
    axes[1, 1].set_xlabel('epoch')
    axes[1, 1].set_ylabel('acc_val')
    axes[1, 1].set_yticks(y_axis_acc)
    axes[1, 1].set_ylim(min(y_axis_acc), max(y_axis_acc))
    axes[1, 1].grid(True)
    for y in np.arange(0.5, 0.71, 0.05):
        axes[1, 1].axhline(y, color='black', linestyle='-', linewidth=0.5) 

    axes[1, 1].plot(best_mse_val_idx, accs_val[best_mse_val_idx], 'rx', markersize=8, 
                    label=f"Best val epoch : acc_val = {accs_val[best_mse_val_idx]:.3f}")
    axes[1, 1].axhline(accs_val[best_mse_val_idx], color='r', linestyle='--')
    axes[1, 1].axvline(best_mse_val_idx, color='r', linestyle='--')
    axes[1, 1].plot(best_acc_val_idx, accs_val[best_acc_val_idx], color='r', marker='*',
                    markersize=8, label=f"Best acc_val : {accs_val[best_acc_val_idx]:.3f}")
    axes[1, 1].legend()

    # Add layout adjustments
    plt.tight_layout()

    # Show plot
    plt.show()



def plot_heat_map(model : nn.Module, dataloader : DataLoader, n_games : int=51, vmax : int=0.25):
    device = next(model.parameters()).device
    
    data_iter = iter(dataloader)
    
    # Get attention maps
    with torch.no_grad():
        kq, v, targets, padding_masks = next(data_iter)
        kq = kq.to(device)
        v = v.to(device)
        targets = targets.to(device)
        padding_masks = padding_masks.to(device)
        _, attn_maps = model(kq, v)

    attn_maps = attn_maps[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    fig.suptitle("Attention Maps of Model Heads")

    sns.heatmap(attn_maps[0, :n_games, :n_games].cpu(), ax=axes[0, 0], cmap='plasma', cbar=True, vmin=0, vmax=vmax)
    axes[0, 0].set_title("Attention Head 1")
    axes[0, 0].set_xlabel("Game attended to")
    axes[0, 0].set_ylabel("Game attending")
    sns.heatmap(attn_maps[1, :n_games, :n_games].cpu(), ax=axes[0, 1], cmap='plasma', cbar=True, vmin=0, vmax=vmax)
    axes[0, 1].set_title("Attention Head 2")
    axes[0, 1].set_xlabel("Game attended to")
    axes[0, 1].set_ylabel("Game attending")
    sns.heatmap(attn_maps[2, :n_games, :n_games].cpu(), ax=axes[1, 0], cmap='plasma', cbar=True, vmin=0, vmax=vmax)
    axes[1, 0].set_title("Attention Head 3")
    axes[1, 0].set_xlabel("Game attended to")
    axes[1, 0].set_ylabel("Game attending")
    sns.heatmap(attn_maps[3, :n_games, :n_games].cpu(), ax=axes[1, 1], cmap='plasma', cbar=True, vmin=0, vmax=vmax)
    axes[1, 1].set_title("Attention Head 4")
    axes[1, 1].set_xlabel("Game attended to")
    axes[1, 1].set_ylabel("Game attending")
        
    # Add layout adjustments
    plt.tight_layout()

    # Show plot
    plt.show()
    