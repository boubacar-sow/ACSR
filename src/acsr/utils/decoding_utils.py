"""
Decoding utilities for the ACSR system.
This module contains functions for decoding sequences.
"""

import torch
import torch.nn.functional as F


def greedy_decoder(output, blank):
    """
    Greedy decoder for CTC outputs.
    
    Args:
        output (torch.Tensor): Output logits from the model.
        blank (int): Blank token index.
        
    Returns:
        list: List of decoded sequences.
    """
    arg_maxes = torch.argmax(output, dim=2)  # Get the most likely class for each time step
    decodes = []
    for args in arg_maxes:
        args = torch.unique_consecutive(args)  # Remove consecutive repeated indices
        decode = []
        for index in args:
            if index != blank:
                decode.append(index.item())  # Append non-blank and non-repeated tokens
        decodes.append(decode)
    return decodes


def logsumexp(a, b):
    """
    Combine two log values in a numerically stable manner.
    
    Args:
        a (float): First log value.
        b (float): Second log value.
        
    Returns:
        float: Combined log value.
    """
    return torch.logaddexp(torch.tensor(a), torch.tensor(b)).item()


def compute_lm_score(seq, nextsyllable_model, sos_idx, pad_idx, max_seq_len, device):
    """
    Compute language model score for a sequence.
    
    Args:
        seq (list): Input sequence.
        nextsyllable_model (nn.Module): Language model.
        sos_idx (int): Start-of-sequence token index.
        pad_idx (int): Padding token index.
        max_seq_len (int): Maximum sequence length.
        device (torch.device): Device to run the model on.
        
    Returns:
        float: Language model score.
    """
    if not seq:
        return 0.0
    
    lm_score = 0.0
    
    for i in range(1, len(seq)):
        prefix = seq[:i]
        if len(prefix) < max_seq_len:
            padded_prefix = [pad_idx] * (max_seq_len - len(prefix)) + prefix
        else:
            padded_prefix = prefix[-max_seq_len:]
        
        input_tensor = torch.tensor(padded_prefix, dtype=torch.long, device=device).unsqueeze(0)
        lm_logits = nextsyllable_model(input_tensor)  # shape: (1, vocab_size)
        lm_log_probs = F.log_softmax(lm_logits, dim=-1)
        token_log_prob = lm_log_probs[0, seq[i]].item()
        lm_score += token_log_prob
    
    return lm_score 