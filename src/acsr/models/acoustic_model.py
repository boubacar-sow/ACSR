"""
Acoustic model module for the ACSR system.
This module contains the acoustic model classes used for decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerBlock


class ThreeStreamFusionEncoder(nn.Module):
    """
    Three-stream fusion encoder for processing hand shape, hand position, and lip features.
    
    Args:
        hand_shape_dim (int): Dimension of hand shape features.
        hand_pos_dim (int): Dimension of hand position features.
        lips_dim (int): Dimension of lip features.
        visual_lips_dim (int): Dimension of visual lip features.
        hidden_dim (int, optional): Hidden dimension. Defaults to 128.
        n_layers (int, optional): Number of GRU layers. Defaults to 2.
    """
    
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, visual_lips_dim, hidden_dim=128, n_layers=2):
        super(ThreeStreamFusionEncoder, self).__init__()

        self.hand_shape_gru = nn.GRU(hand_shape_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.hand_pos_gru = nn.GRU(hand_pos_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.lips_gru = nn.GRU(lips_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        
        # Fusion GRU: note the input size is 3 streams * 2 (bidirectional)
        self.fusion_gru = nn.GRU(hidden_dim * 6, hidden_dim * 3, n_layers, bidirectional=True, batch_first=True)
        
        # CNN for visual lips (if used)
        if visual_lips_dim is not None:
            self.visual_lips_cnn = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, hidden_dim)),
                nn.Flatten(start_dim=1),
                nn.Linear(128 * hidden_dim, hidden_dim)
            )
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(embed_dim=hidden_dim*6, num_heads=4)
    
    def forward(self, hand_shape, hand_pos, lips, visual_lips=None):
        """
        Forward pass through the encoder.
        
        Args:
            hand_shape (torch.Tensor): Hand shape features.
            hand_pos (torch.Tensor): Hand position features.
            lips (torch.Tensor): Lip features.
            visual_lips (torch.Tensor, optional): Visual lip features. Defaults to None.
            
        Returns:
            torch.Tensor: Encoded features.
        """
        # Process each modality with its respective GRU
        hand_shape_out, _ = self.hand_shape_gru(hand_shape)  # (batch, seq, hidden_dim*2)
        hand_pos_out, _ = self.hand_pos_gru(hand_pos)
        lips_out, _ = self.lips_gru(lips)

        # Concatenate all modalities along the last dimension
        combined_features = torch.cat([hand_shape_out, hand_pos_out, lips_out], dim=-1)
        
        # Apply fusion GRU
        fusion_out, _ = self.fusion_gru(combined_features)
        return fusion_out


class AttentionDecoder(nn.Module):
    """
    Attention decoder for sequence-to-sequence modeling.
    
    Args:
        encoder_dim (int): Dimension of encoder outputs.
        output_dim (int): Dimension of output (vocabulary size).
        hidden_dim_decoder (int, optional): Hidden dimension of decoder. Defaults to None.
        n_layers (int, optional): Number of GRU layers. Defaults to 1.
    """
    
    def __init__(self, encoder_dim, output_dim, hidden_dim_decoder=None, n_layers=1):
        super(AttentionDecoder, self).__init__()
        # If not provided, set hidden_dim_decoder equal to encoder_dim
        if hidden_dim_decoder is None:
            hidden_dim_decoder = encoder_dim
        self.embedding = nn.Embedding(output_dim, hidden_dim_decoder)
        self.gru = nn.GRU(hidden_dim_decoder + encoder_dim, hidden_dim_decoder, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim_decoder, output_dim)
        self.encoder_dim = encoder_dim
        self.hidden_dim_decoder = hidden_dim_decoder

    def forward(self, encoder_outputs, target_seq):
        """
        Forward pass through the decoder.
        
        Args:
            encoder_outputs (torch.Tensor): Outputs from the encoder.
            target_seq (torch.Tensor): Target sequence for teacher forcing.
            
        Returns:
            torch.Tensor: Decoder outputs.
        """
        batch_size, target_len = target_seq.size()
        hidden = None
        outputs = []

        # For each time step in the target sequence (using teacher forcing)
        for t in range(target_len):
            # Get embedding for current target token
            embedded = self.embedding(target_seq[:, t].long()).unsqueeze(1)
            
            # Dot-product attention
            attn_scores = torch.bmm(embedded, encoder_outputs.transpose(1, 2))
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Compute context vector as weighted sum of encoder outputs
            attn_applied = torch.bmm(attn_weights, encoder_outputs)
            
            # Concatenate embedded input and context vector
            gru_input = torch.cat([embedded, attn_applied], dim=2)
            
            # Pass through GRU
            output, hidden = self.gru(gru_input, hidden)
            output = self.out(output.squeeze(1))
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs


class JointCTCAttentionModel(nn.Module):
    """
    Joint CTC-Attention model for sequence prediction.
    
    Args:
        hand_shape_dim (int): Dimension of hand shape features.
        hand_pos_dim (int): Dimension of hand position features.
        lips_dim (int): Dimension of lip features.
        visual_lips_dim (int): Dimension of visual lip features.
        output_dim (int): Dimension of output (vocabulary size).
        hidden_dim (int, optional): Hidden dimension. Defaults to 128.
    """
    
    def __init__(self, hand_shape_dim, hand_pos_dim, lips_dim, visual_lips_dim, output_dim, hidden_dim=128):
        super(JointCTCAttentionModel, self).__init__()
        self.encoder = ThreeStreamFusionEncoder(hand_shape_dim, hand_pos_dim, lips_dim, visual_lips_dim, hidden_dim)
        self.attention_decoder = AttentionDecoder(encoder_dim=hidden_dim * 6, output_dim=output_dim)
        self.ctc_fc = nn.Linear(hidden_dim * 6, output_dim)
    
    def forward(self, hand_shape, hand_pos, lips, visual_lips=None, target_seq=None):
        """
        Forward pass through the joint model.
        
        Args:
            hand_shape (torch.Tensor): Hand shape features.
            hand_pos (torch.Tensor): Hand position features.
            lips (torch.Tensor): Lip features.
            visual_lips (torch.Tensor, optional): Visual lip features. Defaults to None.
            target_seq (torch.Tensor, optional): Target sequence for teacher forcing. Defaults to None.
            
        Returns:
            tuple: (ctc_logits, attn_logits)
        """
        encoder_out = self.encoder(hand_shape, hand_pos, lips, visual_lips)
        ctc_logits = self.ctc_fc(encoder_out)
        if target_seq is not None:
            attn_logits = self.attention_decoder(encoder_out, target_seq)
        else:
            attn_logits = None
        return ctc_logits, attn_logits 