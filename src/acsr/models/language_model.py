"""
Language model module for the ACSR system.
This module contains the language model classes used for next syllable prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NextSyllableLSTM(nn.Module):
    """
    LSTM-based language model for next syllable prediction.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
        hidden_dim (int): Dimension of the hidden state.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers,
            dropout=dropout, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        """
        Forward pass through the language model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, vocab_size).
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :]) 