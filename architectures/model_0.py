#### STD IMPORTS
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

#### EXTERNAL IMPORTS
# TODO: make it so this is handled in another file.
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayerA(nn.Module):
    def __init__(self, 
                 input_dim,  
                 embed_dim,
                 n_heads=1,
                 dropout=0.1
    ):
        super(TransformerLayerA, self).__init__()

        self.W_k = nn.Linear(input_dim, embed_dim)
        self.W_q = nn.Linear(input_dim, embed_dim)
        self.W_v = nn.Linear(input_dim, embed_dim)
        self.n_heads = n_heads
        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, kq, v) -> torch.Tensor:
        # FOR TIME BEING:
        #  * keys are OHE matchups
        #  * values are stats
        
        X = torch.concat((kq, v), dim=-1)
        n, T, _ = kq.shape
        n_heads = self.n_heads
        dropout = self.dropout
        
        keys = self.W_k(X)
        queries = self.W_q(X)
        values = self.W_v(X)

        keys = keys.view(n, T, n_heads, -1).transpose(1, 2).contiguous()
        queries = queries.view(n, T, n_heads, -1).transpose(1, 2).contiguous()
        values = values.view(n, T, n_heads, -1).transpose(1, 2).contiguous()
        
        # Causal mask, cannot attend to diagonal
        # Test; allowing the first game to attend to itself
        attn_weights = queries @ keys.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)
        attn_scores = F.softmax(attn_weights + mask, dim=-1)
        
        # Apply attention: v' = A @ v
        attn_out = attn_scores @ values
        attn_out = F.dropout(attn_out, dropout)
        attn_out = attn_out.transpose(2, 1).contiguous().view(n, T, -1)
        values = values.transpose(2, 1).contiguous().view(n, T, -1)
        values = self.norm1(values + attn_out)
        
        # Apply feed-forward
        ff_out = self.ff_layer(values)
        ff_out = F.dropout(ff_out, dropout)
        values = self.norm2(values + ff_out)
        return values

class TransformerLayerB(nn.Module):
    def __init__(self, 
                input_dim_kq,  
                input_dim_v,
                embed_dim_kq,
                embed_dim_v,
                n_heads=1,
                dropout=0.1):
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
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(embed_dim_v)
        self.norm2 = nn.LayerNorm(embed_dim_v)

    def forward(self, kq, v) -> torch.Tensor:
        # FOR TIME BEING:
        #  * keys are OHE matchups
        #  * values are stats
        n, T, _ = kq.shape
        n_heads = self.n_heads
        dropout = self.dropout

        keys = self.W_k(kq)
        queries = self.W_q(kq)
        values = self.W_v(v)

        keys = keys.view(n, T, n_heads, -1).transpose(1, 2).contiguous()
        queries = queries.view(n, T, n_heads, -1).transpose(1, 2).contiguous()
        values = values.view(n, T, n_heads, -1).transpose(1, 2).contiguous()
        
        # Causal mask, cannot attend to diagonal
        # Test; allowing the first game to attend to itself
        attn_weights = queries @ keys.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=0)
        mask[0,0] = 0 # To get rid of NaN's
        attn_scores = F.softmax(attn_weights + mask, dim=-1)
        
        # Apply attention: v' = A @ v
        attn_out = attn_scores @ values
        attn_out = F.dropout(attn_out, dropout)
        attn_out = attn_out.transpose(2, 1).contiguous().view(n, T, -1)
        values = values.transpose(2, 1).contiguous().view(n, T, -1)
        #values = self.norm1(values + attn_out)
        values = attn_out
        
        # Apply feed-forward
        ff_out = self.ff_layer(values)
        ff_out = F.dropout(ff_out, dropout)
        #values = self.norm2(values + ff_out)
        values = ff_out
        return values

class Model(nn.Module):
    def __init__(self, 
                 input_dim_kq,
                 input_dim_v,  
                 embed_dim_kq,
                 embed_dim_v,
                 output_dim,
                 n_heads=1,
                 dropout=0.1
    ):
        super(Model, self).__init__()

        self.n_heads = n_heads
        
        # Uses all data as keys, queries, values
        self.transformer_layer_A = TransformerLayerA(
            input_dim=input_dim_kq + input_dim_v,
            embed_dim=embed_dim_v,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Uses specified kq as keys/queries
        # Uses all data as values (output from A)
        self.transformer_layer_B = TransformerLayerB(
            input_dim_kq=input_dim_kq, 
            embed_dim_kq=embed_dim_kq,
            input_dim_v=embed_dim_v,
            embed_dim_v=embed_dim_v,
            n_heads=n_heads,
            dropout=dropout
        )
        self.output_layer = nn.Linear(embed_dim_v, output_dim)
        self.dropout=dropout

    def forward(self, kq, v):
        v = self.transformer_layer_A(kq, v)
        v = self.transformer_layer_B(kq, v)
        return self.output_layer(v)
    