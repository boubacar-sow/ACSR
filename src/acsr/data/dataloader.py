"""
Dataloader module for the ACSR system.
This module contains dataloader-related functions.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def custom_collate_fn(batch):
    """
    Custom collate function for padding sequences in a batch.
    
    Args:
        batch (list): List of (hand_shape, hand_pos, acoustic_lips, labels) tuples.
        
    Returns:
        tuple: Tuple of padded tensors.
    """
    hand_shape, hand_pos, acoustic_lips, labels = zip(*batch)
    
    # Pad sequences for hand shape, hand position, and acoustic lips
    hand_shape_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in hand_shape], batch_first=True, padding_value=0)
    hand_pos_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in hand_pos], batch_first=True, padding_value=0)
    acoustic_lips_padded = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in acoustic_lips], batch_first=True, padding_value=0)
    
    # Pad labels
    labels_padded = pad_sequence([torch.tensor(y, dtype=torch.long) for y in labels], batch_first=True, padding_value=0)  # Use appropriate padding value
    
    return hand_shape_padded, hand_pos_padded, acoustic_lips_padded, labels_padded


def data_to_dataloader(data, batch_size=4, shuffle=True, num_workers=6):
    """
    Convert data dictionary to a DataLoader.
    
    Args:
        data (dict): Dictionary containing data arrays.
        batch_size (int, optional): Batch size. Defaults to 4.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of worker processes. Defaults to 6.
        
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = list(zip(
        data['X_acoustic_hand_shape'],
        data['X_acoustic_hand_pos'],
        data['X_acoustic_lips'],
        data['y']
    ))
    
    # Create a DataLoader with the custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    return dataloader 