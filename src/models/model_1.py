#### STD IMPORTS
import sys
import os
from typing import Dict

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

#### EXTERNAL IMPORTS
# TODO: make it so this is handled in another file.
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_dim_kq=64,
        embed_dim_kq=128,
        input_dim_v=1, 
        embed_dim_v=4,
        n_heads=4,
        dropout=0.1
    ):
        super(TransformerLayer, self).__init__()
        self.W_k = nn.Linear(input_dim_kq, embed_dim_kq)
        self.W_q = nn.Linear(input_dim_kq, embed_dim_kq)
        self.W_v = nn.Linear(input_dim_v, embed_dim_v)
        self.n_heads = n_heads
        
    def forward(self, kq, v):
        n, T, _ = v.shape
        n_heads = self.n_heads

        ohe_1 = kq[:, :, 0]
        ohe_2 = kq[:, :, 1]
        keys = self.W_k(torch.concat((ohe_1, ohe_2), dim=-1))
        queries = self.W_q(torch.concat((ohe_1, ohe_2), dim=-1))
        values = self.W_v(v)
        
        keys = keys.view(n, T, n_heads, -1).transpose(1, 2)
        queries = queries.view(n, T, n_heads, -1).transpose(1, 2)
        values = values.view(n, T, n_heads, -1).transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(-1, -2)
        causal_mask = torch.triu(torch.ones(T, T), diagonal=0).to(attn_scores.device)
        causal_mask[0,0] = 0 # To avoid NaN (bandaid fix)
        causal_mask[causal_mask == 1] = float('-inf')
        attn_weights = F.softmax((attn_scores + causal_mask), dim=-1)
        
        values = attn_weights @ values
        values = values.transpose(2, 1).view(n, T, -1)
        return values
    
class Model(nn.Module):
    def __init__(
        self, 
        input_dim_kq=64,
        embed_dim_kq=128,
        input_dim_v=1,
        embed_dim_v=4,
        n_heads=4,
        dropout=0.1,
        output_dim=1
    ):
        super(Model, self).__init__()
        self.transformer_layer = TransformerLayer(
            input_dim_kq,
            embed_dim_kq,
            input_dim_v,
            embed_dim_v,
            n_heads,
            dropout
        )
        self.output_layer = nn.Linear(embed_dim_v, output_dim)
    
    def forward(self, batch : Dict[str, torch.Tensor]):
        kq = batch.get('kq')
        v = batch.get('v')
        assert kq is not None and v is not None, "Missing required keys in input dictionary"
        v = self.transformer_layer(kq, v)
        v = self.output_layer(v)
        return v
