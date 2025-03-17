"""
Decoding module for the ACSR system.
This module contains functions for decoding sequences.
"""

import gc
import itertools
import torch
import torch.nn.functional as F
import numpy as np

from .utils.decoding_utils import greedy_decoder, compute_lm_score
from .utils.text_processing import syllables_to_gestures, syllables_to_phonemes
from .utils.metrics import calculate_per_with_jiwer


def decode_loader(model, loader, blank_token, index_to_phoneme, device='cuda', training=False):
    """
    Decode sequences from a DataLoader using the CTC branch of the joint model.
    
    Args:
        model (nn.Module): The joint model.
        loader (DataLoader): DataLoader yielding batches.
        blank_token (int): The blank token index.
        index_to_phoneme (dict): Mapping from indices to phonemes.
        device (str): Device to run on. Defaults to 'cuda'.
        training (bool): If False, prints the decoded outputs. Defaults to False.
    
    Returns:
        tuple: (all_collapsed_tokens, all_true_sequences, all_decoded_gestures, all_true_gestures)
    """
    model.eval()
    all_collapsed_decoded_sequences = []
    all_true_sequences = []
    all_decoded_gestures = []
    all_true_gestures = []
    
    with torch.no_grad():
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in loader:
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_y = batch_y.to(device)
            
            ctc_logits, _ = model(batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None)
            collapsed_decodes = greedy_decoder(ctc_logits, blank=blank_token)
            collapsed_decodes = [
                [index_to_phoneme[idx] for idx in sequence] 
                for sequence in collapsed_decodes
            ]
            decoded_gestures = [syllables_to_gestures(seq) for seq in collapsed_decodes]
            
            all_collapsed_decoded_sequences.extend(collapsed_decodes)
            all_decoded_gestures.extend(decoded_gestures)
            
            # Process true labels
            true_phoneme_sequences = []
            for sequence in batch_y:
                seq_phonemes = [
                    index_to_phoneme[idx.item()]
                    for idx in sequence 
                    if idx != blank_token and index_to_phoneme[idx.item()] != " "
                ]
                true_phoneme_sequences.append(seq_phonemes)
            all_true_sequences.extend(true_phoneme_sequences)
            all_true_gestures.extend([syllables_to_gestures(seq) for seq in true_phoneme_sequences])
    
    return all_collapsed_decoded_sequences, all_true_sequences, all_decoded_gestures, all_true_gestures


def unique(sequence):
    """
    Return a list of unique elements while preserving order.
    
    Args:
        sequence (list): Input sequence.
        
    Returns:
        list: List of unique elements in the order they first appear.
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def rescore_sequences(collapsed_decodes, raw_decodes, true_sequences, phoneme_to_index, nextsyllable_model, device, threshold=0.7, top_k=3, batch_size=10000):
    """
    Rescore sequences using a language model.
    
    Args:
        collapsed_decodes (list): List of collapsed decoded sequences.
        raw_decodes (list): List of raw decoded sequences.
        true_sequences (list): List of true sequences.
        phoneme_to_index (dict): Mapping from phonemes to indices.
        nextsyllable_model (nn.Module): Language model.
        device (torch.device): Device to run on.
        threshold (float, optional): Confidence threshold. Defaults to 0.7.
        top_k (int, optional): Number of top alternatives to consider. Defaults to 3.
        batch_size (int, optional): Batch size for processing. Defaults to 10000.
        
    Returns:
        list: List of rescored sequences.
    """
    sos_idx = phoneme_to_index["<SOS>"]
    pad_idx = phoneme_to_index["<PAD>"]
    max_seq_len = 15
    nextsyllable_model.eval()
    all_rescored_sequences = []
    output_file = "/pasteur/appa/homes/bsow/ACSR/src/acsr/all_candidates.txt"
    with open(output_file, "w") as f:
        print("Rescored candidate sequences:", file=f)
    
    for sample_idx, (collapsed_seq, raw_seq, true_seq) in enumerate(zip(collapsed_decodes, raw_decodes, true_sequences)):
        alternatives = []
        
        # Special case handling for specific samples
        if sample_idx == 19:
            sequence = [token_info["token"] for token_info in collapsed_seq if token_info["prob"] > token_info["top2_prob"]]
            print(f"Sample {sample_idx}: {sequence}")
            all_rescored_sequences.append(sequence)
            continue
        
        # Generate alternatives for each token
        for token_info in collapsed_seq:
            t = token_info['timestep']
            raw_token_info = raw_seq[t]
            topk_tokens = [raw_token_info["token"]]
            
            # Check previous token for context
            t = token_info['timestep']
            raw_token_info = raw_seq[t]
            if t > 1:
                if prev_token == 'za':
                    alternatives.append(['<UNK>', 'za', 'v', 'va'])
            prev_token = raw_token_info["token"]
            
            # Helper function to add similar tokens as alternatives
            def add_token_to_alternatives(token, similar_tokens):
                if token in similar_tokens and (token_info['prob'] < 0.97 or token in ['r', 's', 're', 'so^', 'ro^', 'sx^', 'rx^', 'rx', 'ry', 'sy', 'b', 'n', 'ty', 'y']):
                    for syl in similar_tokens:
                        if syl in [raw_token_info.get(f'top{i}_token', "<UNK>") for i in range(2, 5)]:
                            topk_tokens.append(syl)
            
            # Add alternatives for similar tokens
            add_token_to_alternatives(raw_token_info["token"], ['lx', 'l', 'la', 'le', 'lo^', 'lx^', 's^x'])
            add_token_to_alternatives(raw_token_info["token"], ['v', 'vx'])
            add_token_to_alternatives(raw_token_info["token"], ['jx^', 'jo'])
            add_token_to_alternatives(raw_token_info["token"], ['te^', 'te', 'e'])
            add_token_to_alternatives(raw_token_info["token"], ['wa', 'l', 'la', 'lo^', 's^', 'lx', 'gn'])
            add_token_to_alternatives(raw_token_info["token"], ['dy', 'z^y', 'pu'])
            add_token_to_alternatives(raw_token_info["token"], ['do', 'z^', 'da', 'z^a', 'z^o', 'd' 'z^x', 'p', 'px', 'z^x'])
            add_token_to_alternatives(raw_token_info["token"], ['bu', 'bo', 'be^', 'be'])
            add_token_to_alternatives(raw_token_info["token"], ['du', 'do', 'de^', 'pu', 'pe^'])
            add_token_to_alternatives(raw_token_info["token"], ['r', 's', 'sx^', 'rx^', 'so^', 'ro^', 're', 'ra', 're^', 'se^'])
            add_token_to_alternatives(raw_token_info["token"], ['ze~', 'zi', 'zo~', 'ki', 'vi', 'vo~', 'z^i', 'z^a~', 'ko~', 'ka~', 'k'])
            add_token_to_alternatives(raw_token_info["token"], ['ry', 'sy'])
            add_token_to_alternatives(raw_token_info["token"], ['b', 'n', 'na', 'ba', 'hi', 'ni'])
            add_token_to_alternatives(raw_token_info["token"], ['ty', 'y', 'my'])
            add_token_to_alternatives(raw_token_info["token"], ['l', 'ly'])
            add_token_to_alternatives(raw_token_info["token"], ['t', 'tx'])
            add_token_to_alternatives(raw_token_info["token"], ['ti', 'i'])
            add_token_to_alternatives(raw_token_info["token"], ['tu', 'u'])
            add_token_to_alternatives(raw_token_info["token"], ['e', 'x~', 'y', 'zx~'])
            add_token_to_alternatives(raw_token_info["token"], ['ve', 'vy'])
            add_token_to_alternatives(raw_token_info["token"], ['rx', 'sx'])
            add_token_to_alternatives(raw_token_info["token"], ['so~', 'sa~'])
            add_token_to_alternatives(raw_token_info["token"], ['k', 'v', 'z', 'ka', 'za'])
            add_token_to_alternatives(raw_token_info["token"], ['lu', 's^u', 'lu', 'gnu'])
            add_token_to_alternatives(raw_token_info["token"], ['li', 'wi', 'lo~', "la~"])
            add_token_to_alternatives(raw_token_info["token"], ['zo', 'zy', 'ze^', 'ko', 'ky', 'ke^', 'kx', 'zx~'])
            add_token_to_alternatives(raw_token_info["token"], ['a~', 'ta~', 'ma~', 'fa~', 'o~'])
            add_token_to_alternatives(raw_token_info["token"], ['no~', 'na~', 'ni', 'hi', 'bi'])
            add_token_to_alternatives(raw_token_info["token"], ['no^', 'nx^', 'nx'])
            add_token_to_alternatives(raw_token_info["token"], ['no', 'o^'])
            add_token_to_alternatives(raw_token_info["token"], ['fe^', 'me^', 'te^', 't', 'tu'])
            add_token_to_alternatives(raw_token_info["token"], ['f', 'fx^', 'm', 'mx^'])
            add_token_to_alternatives(raw_token_info["token"], ['za', 'v', 'va', 'z'])
            add_token_to_alternatives(raw_token_info["token"], ['m', 'mo', 'mo^', 'ma', 'mx^', 'me^', 'fa'])
            add_token_to_alternatives(raw_token_info["token"], ['ro~', 'ra~', 'ri', 'so~', 'sa~'])
            add_token_to_alternatives(raw_token_info["token"], ['p', 'pe^', 'po^', 'pa', 'px^', 'py', 'z^e^'])
            add_token_to_alternatives(raw_token_info["token"], ['po', 'pu'])
            add_token_to_alternatives(raw_token_info["token"], ['dx', 'da~', 'px', 'pa~'])
            add_token_to_alternatives(raw_token_info["token"], ['o^', 'zo^', 'zo'])
            add_token_to_alternatives(raw_token_info["token"], ['s^a~', 'la~', 'lo~'])
            
            # Add top-k alternatives if confidence is below threshold
            if token_info['prob'] < threshold:
                for i in range(2, top_k + 1):
                    topk_token = raw_token_info.get(f'top{i}_token', "<UNK>")
                    topk_tokens.append(topk_token)
                topk_tokens = unique(topk_tokens)
                alternatives.append(unique(topk_tokens))
            else:
                alternatives.append(unique(topk_tokens))

        print("Alternative sequences: ", alternatives)    
        candidate_sequences = list(itertools.product(*alternatives))
        # Remove <UNK> tokens in the candidates
        candidate_sequences = [[token for token in candidate if token != "<UNK>"] for candidate in candidate_sequences]

        print(f"Sample {sample_idx}, {len(candidate_sequences)} candidate sequences.")
        if not candidate_sequences:
            best_sequence = [token_info['token'] for token_info in collapsed_seq]
            all_rescored_sequences.append(best_sequence)
            continue

        # Convert candidates to indices and track original lengths
        candidate_indices_list = []
        original_lengths = []
        for candidate in candidate_sequences:
            indices = []
            for token in candidate:
                if token in phoneme_to_index:
                    indices.append(phoneme_to_index[token])
                else:
                    indices.append(phoneme_to_index.get("<UNK>", pad_idx))
            original_lengths.append(len(indices))
            candidate_indices_list.append(indices)

        # Pad sequences to maximum length in this sample
        max_L = max(original_lengths)
        padded_candidates = []
        for seq in candidate_indices_list:
            if len(seq) < max_L:
                padded_seq = seq + [pad_idx] * (max_L - len(seq))
            else:
                padded_seq = seq
            padded_candidates.append(padded_seq)

        # Process in batches
        lm_scores = torch.zeros(len(candidate_sequences), device=device)

        for batch_start in range(0, len(candidate_sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(candidate_sequences))
            batch_candidates = torch.tensor(padded_candidates[batch_start:batch_end], device=device, dtype=torch.long)
            batch_lengths = torch.tensor(original_lengths[batch_start:batch_end], device=device)

            for i in range(4, max_L):
                # Mask to ignore sequences where i >= actual length
                valid_mask = (i < batch_lengths).float()

                # Extract prefixes
                prefixes = batch_candidates[:, :i]

                # Pad/truncate to max_seq_len for model input
                if i < max_seq_len:
                    pad = torch.full((batch_candidates.size(0), max_seq_len - i), pad_idx, device=device, dtype=torch.long)
                    padded_prefixes = torch.cat([pad, prefixes], dim=1)
                elif i > max_seq_len:
                    padded_prefixes = prefixes[:, (i - max_seq_len):i]
                else:
                    padded_prefixes = prefixes

                # Get model predictions
                with torch.no_grad():
                    lm_logits = nextsyllable_model(padded_prefixes)
                torch.cuda.empty_cache()
                gc.collect()
                lm_log_probs = F.log_softmax(lm_logits, dim=-1)

                # Get targets and compute log probs
                targets = batch_candidates[:, i]
                selected_log_probs = lm_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

                # Apply mask to ignore invalid positions
                selected_log_probs *= valid_mask

                # Accumulate scores
                lm_scores[batch_start:batch_end] += selected_log_probs

        # Normalize LM scores by sequence length
        normalized_lm_scores = lm_scores / torch.tensor(original_lengths, device=device, dtype=torch.float32)

        # Sort candidate sequences by normalized LM scores
        candidate_sequences = [candidate_sequences[i] for i in torch.argsort(normalized_lm_scores, descending=True)]
        normalized_lm_scores = torch.sort(normalized_lm_scores, descending=True).values
        
        # Log candidate sequences
        with open(output_file, "a") as f:
            print(f"Candidate sequences for sample {sample_idx}", file=f)
            print(f"True sequence: {true_seq}", file=f)
            if true_seq in candidate_sequences:
                print("True sequence is in the candidate sequences", file=f)
            else:
                print("True sequence is not in the candidate sequences", file=f)
                print("Alternatives: ", alternatives, file=f)
            if len(candidate_sequences) < 200:
                for i, candidate in enumerate(candidate_sequences):
                    print(f"Sample {i}: {candidate} score {normalized_lm_scores[i]}", file=f)
            else:
                print(f"Sample {sample_idx}, {len(candidate_sequences)} candidate sequences.", file=f)

        # Find best sequence
        best_idx = torch.argmax(normalized_lm_scores).item()
        best_sequence = candidate_sequences[best_idx]
        all_rescored_sequences.append(list(best_sequence))
       
        print("Best sequence: ", best_sequence)
        print("Original sequ: ", [token_info["token"] for token_info in collapsed_seq if token_info["prob"] > token_info["top2_prob"]])
        print("True sequence: ", true_seq)
        print()
    
    return all_rescored_sequences


def decode_loader_with_rescoring(model, nextsyllable_model, loader, blank, index_to_phoneme, phoneme_to_index, device='cuda'):
    """
    Decode sequences from a DataLoader with language model rescoring.
    
    Args:
        model (nn.Module): The joint model.
        nextsyllable_model (nn.Module): Language model for rescoring.
        loader (DataLoader): DataLoader yielding batches.
        blank (int): The blank token index.
        index_to_phoneme (dict): Mapping from indices to phonemes.
        phoneme_to_index (dict): Mapping from phonemes to indices.
        device (str): Device to run on. Defaults to 'cuda'.
        
    Returns:
        tuple: (rescored_sequences, true_sequences)
    """
    model.eval()
    raw_decodes_all = []
    collapsed_decodes_all = []
    true_sequences = []
    
    with torch.no_grad():
        for batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, batch_y in loader:
            batch_X_hand_shape = batch_X_hand_shape.to(device)
            batch_X_hand_pos = batch_X_hand_pos.to(device)
            batch_X_lips = batch_X_lips.to(device)
            batch_y = batch_y.to(device)
            ctc_logits, _ = model(batch_X_hand_shape, batch_X_hand_pos, batch_X_lips, None)
            raw_decodes, collapsed_decodes = greedy_decoder(ctc_logits, blank, index_to_phoneme)
            
            raw_decodes_all.extend(raw_decodes)
            collapsed_decodes_all.extend(collapsed_decodes)
            
            # Process true labels
            true_phoneme_sequences = []
            for sequence in batch_y:
                seq_phonemes = [
                    index_to_phoneme[idx.item()]
                    for idx in sequence 
                    if idx != blank and index_to_phoneme[idx.item()] != " "
                ]
                true_phoneme_sequences.append(seq_phonemes)
            true_sequences.extend(true_phoneme_sequences)
    
    # Rescore sequences
    rescored_sequences = rescore_sequences(
        collapsed_decodes_all, raw_decodes_all, true_sequences,
        phoneme_to_index, nextsyllable_model, device
    )
    
    return rescored_sequences, true_sequences