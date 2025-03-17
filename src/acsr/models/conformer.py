"""
Conformer module for the ACSR system.
This module contains the ConformerBlock class used in the acoustic model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerBlock(nn.Module):
    """
    Conformer block combining self-attention and convolution for sequence modeling.
    
    Args:
        dim (int): The input dimension.
        heads (int, optional): Number of attention heads. Defaults to 4.
        ff_mult (int, optional): Feed-forward dimension multiplier. Defaults to 4.
        conv_expansion_factor (int, optional): Convolution expansion factor. Defaults to 2.
        conv_kernel_size (int, optional): Convolution kernel size. Defaults to 31.
        attn_dropout (float, optional): Attention dropout rate. Defaults to 0.0.
        ff_dropout (float, optional): Feed-forward dropout rate. Defaults to 0.0.
        conv_dropout (float, optional): Convolution dropout rate. Defaults to 0.0.
    """
    
    def __init__(
        self,
        dim,
        heads=4,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
    ):
        super().__init__()
        
        self.ff1 = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = SelfAttention(dim, heads=heads, dropout=attn_dropout)
        self.conv = ConformerConvModule(
            dim, expansion_factor=conv_expansion_factor, 
            kernel_size=conv_kernel_size, dropout=conv_dropout
        )
        self.ff2 = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Forward pass through the Conformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        # First feed-forward module
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention module
        x = x + self.attn(x)
        
        # Convolution module
        x = x + self.conv(x)
        
        # Second feed-forward module
        x = x + 0.5 * self.ff2(x)
        
        # Final layer normalization
        return self.norm(x)


class FeedForward(nn.Module):
    """
    Feed-forward module with layer normalization and residual connection.
    
    Args:
        dim (int): Input dimension.
        mult (int, optional): Hidden dimension multiplier. Defaults to 4.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """
    
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Forward pass through the feed-forward module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        return self.net(self.norm(x))


class SelfAttention(nn.Module):
    """
    Self-attention module with layer normalization.
    
    Args:
        dim (int): Input dimension.
        heads (int, optional): Number of attention heads. Defaults to 4.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """
    
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x):
        """
        Forward pass through the self-attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        x = self.norm(x)
        return self.attn(x, x, x, need_weights=False)[0]


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module with layer normalization.
    
    Args:
        dim (int): Input dimension.
        expansion_factor (int, optional): Expansion factor for pointwise conv. Defaults to 2.
        kernel_size (int, optional): Kernel size for depthwise conv. Defaults to 31.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """
    
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        
        inner_dim = dim * expansion_factor
        padding = (kernel_size - 1) // 2
        
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Forward pass through the convolution module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        x = self.norm(x)
        
        # Transpose for 1D convolution
        x = x.transpose(1, 2)
        x = self.net(x)
        
        # Transpose back
        return x.transpose(1, 2) 