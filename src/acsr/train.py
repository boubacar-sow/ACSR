"""
Training module for the ACSR system.
This module contains functions for training models.
"""

import time
import sys
import torch
import torch.nn.functional as F
import wandb

from .utils.metrics import calculate_per_with_jiwer
from .utils.decoding_utils import greedy_decoder


def joint_ctc_attention_loss(ctc_logits, attn_logits, target_seq, input_lengths, label_lengths, alpha, device, blank_idx):
    """
    Compute joint CTC-Attention loss.
    
    Args:
        ctc_logits (torch.Tensor): CTC logits from the model.
        attn_logits (torch.Tensor): Attention logits from the model.
        target_seq (torch.Tensor): Target sequence.
        input_lengths (torch.Tensor): Lengths of input sequences.
        label_lengths (torch.Tensor): Lengths of label sequences.
        alpha (float): Weight for CTC loss.
        device (torch.device): Device to run on.
        blank_idx (int): Blank token index.
        
    Returns:
        tuple: (total_loss, ctc_loss, attn_loss)
    """
    # CTC loss branch
    log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
    ctc_loss = F.ctc_loss(
        log_probs,
        target_seq,       # Padded labels (expected as LongTensor)
        input_lengths,
        label_lengths,
        blank=blank_idx,
    )
    
    # Attention branch loss
    attn_loss = F.cross_entropy(
        attn_logits.view(-1, attn_logits.size(-1)),
        target_seq.view(-1),
        ignore_index=blank_idx
    )
    
    total_loss = alpha * ctc_loss + (1 - alpha) * attn_loss
    return total_loss, ctc_loss, attn_loss


def validate_model(model, val_loader, alpha, device, blank_idx, index_to_phoneme):
    """
    Validate the model on a validation set.
    
    Args:
        model (nn.Module): Model to validate.
        val_loader (DataLoader): Validation data loader.
        alpha (float): Weight for CTC loss.
        device (torch.device): Device to run on.
        blank_idx (int): Blank token index.
        index_to_phoneme (dict): Mapping from indices to phonemes.
        
    Returns:
        tuple: (avg_val_loss, avg_ctc_loss, avg_attn_loss, val_per)
    """
    model.eval()
    val_loss = 0.0
    total_ctc_loss = 0.0
    total_attn_loss = 0.0

    all_decoded_sequences = []
    all_true_sequences = []

    with torch.no_grad():
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in val_loader:
            # Ensure batch_y is of type long
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_y = batch_y.long().to(device)
            
            # Forward pass with teacher forcing
            ctc_logits, attn_logits = model(
                batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None, target_seq=batch_y
            )
            
            # Compute input_lengths and label_lengths
            input_lengths = torch.full(
                (batch_X_hand_shape.size(0),),
                ctc_logits.size(1),
                dtype=torch.long,
                device=device
            )
            label_lengths = (batch_y != blank_idx).sum(dim=1).to(device)
            
            # Compute combined loss
            loss, ctc_loss, attn_loss = joint_ctc_attention_loss(
                ctc_logits, attn_logits, batch_y, input_lengths, label_lengths, alpha, device, blank_idx
            )
            val_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            total_attn_loss += attn_loss.item()

            # Decode for PER calculation
            decoded_batch = greedy_decoder(ctc_logits, blank=blank_idx)
            decoded_sequences = [[index_to_phoneme[idx] for idx in seq] for seq in decoded_batch]
            all_decoded_sequences.extend(decoded_sequences)
            
            # Process true labels
            for sequence in batch_y:
                true_seq = [index_to_phoneme[idx.item()] for idx in sequence if idx != blank_idx]
                all_true_sequences.append(true_seq)

    avg_val_loss = val_loss / len(val_loader)
    avg_ctc_loss = total_ctc_loss / len(val_loader)
    avg_attn_loss = total_attn_loss / len(val_loader)
    
    # Calculate PER
    val_per = calculate_per_with_jiwer(all_decoded_sequences, all_true_sequences)
    
    return avg_val_loss, avg_ctc_loss, avg_attn_loss, val_per


def train_model(model, train_loader, val_loader, optimizer, num_epochs, alpha, device, blank_idx, index_to_phoneme, model_save_path):
    """
    Train the model.
    
    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        num_epochs (int): Number of epochs to train for.
        alpha (float): Weight for CTC loss.
        device (torch.device): Device to run on.
        blank_idx (int): Blank token index.
        index_to_phoneme (dict): Mapping from indices to phonemes.
        model_save_path (str): Path to save the model.
        
    Returns:
        tuple: (best_val_per, best_epoch)
    """
    # Initialize variables to track the best validation PER
    best_val_per = float('inf')  # Start with a very high value
    best_epoch = -1  # Track the epoch where the best model was saved
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_attn_loss = 0.0
        
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in train_loader:
            # Convert targets to LongTensor and move data to device
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_y = batch_y.long().to(device)
            
            # Forward pass with teacher forcing
            ctc_logits, attn_logits = model(
                batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None, target_seq=batch_y
            )
            
            # Compute input_lengths and label_lengths
            input_lengths = torch.full(
                (batch_X_hand_shape.size(0),),
                ctc_logits.size(1),
                dtype=torch.long,
                device=device
            )
            label_lengths = (batch_y != blank_idx).sum(dim=1).to(device)
            
            # Compute joint loss
            loss, ctc_loss, attn_loss = joint_ctc_attention_loss(
                ctc_logits, attn_logits, batch_y, input_lengths, label_lengths, alpha, device, blank_idx
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ctc_loss += ctc_loss.item()
            epoch_attn_loss += attn_loss.item()
        
        # Logging to W&B (or print)
        train_loss_avg = epoch_loss / len(train_loader)
        wandb.log({"train_loss": train_loss_avg, "epoch": epoch + 1})
        wandb.log({"train_ctc_loss": epoch_ctc_loss / len(train_loader), "epoch": epoch + 1})
        wandb.log({"train_attn_loss": epoch_attn_loss / len(train_loader), "epoch": epoch + 1})
        
        # Validate the model
        val_loss, val_ctc_loss, val_attn_loss, val_per = validate_model(
            model, val_loader, alpha, device, blank_idx, index_to_phoneme
        )
        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
        wandb.log({"val_ctc_loss": val_ctc_loss, "epoch": epoch + 1})
        wandb.log({"val_attn_loss": val_attn_loss, "epoch": epoch + 1})
        wandb.log({"val_per": val_per, "epoch": epoch + 1})
        wandb.log({"val_accuracy": 1 - val_per, "epoch": epoch + 1})
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {round(train_loss_avg, 3)}, "
              f"Val Loss: {round(val_loss, 3)}, Accuracy (1-PER): {round(1 - val_per, 3)}, "
              f"Time: {round(time.time() - epoch_start_time, 2)} sec")
        
        # Save the model if it achieves the best validation PER
        if val_per < best_val_per:
            best_val_per = val_per
            best_epoch = epoch + 1
            # Save the model checkpoint
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved at epoch {best_epoch} with Val PER: {round(best_val_per, 3)}")
        
        sys.stdout.flush()
    
    print(f"Training complete. Best model saved at epoch {best_epoch} with Val PER: {round(best_val_per, 3)}")
    return best_val_per, best_epoch