"""
Metrics utilities for the ACSR system.
This module contains functions for computing evaluation metrics.
"""

import jiwer


def calculate_per_with_jiwer(decoded_sequences, true_sequences):
    """
    Calculate Phoneme Error Rate (PER) using jiwer.
    
    Args:
        decoded_sequences (list): List of decoded sequences.
        true_sequences (list): List of true sequences.
        
    Returns:
        float: Phoneme Error Rate.
    """
    # Convert phoneme sequences to space-separated strings
    decoded_str = [" ".join(seq) for seq in decoded_sequences]
    true_str = [" ".join(seq) for seq in true_sequences]
    
    # Calculate PER using jiwer
    try:
        per = jiwer.wer(true_str, decoded_str)
    except Exception as e:
        print("Error calculating PER:", e)
        print("True sequences:", true_str)
        print("Decoded sequences:", decoded_str)
        per = 1.0  # Default to worst case
    
    return per


def compute_edit_distance(ref_words, hyp_words):
    """
    Compute edit distance between reference and hypothesis words.
    
    Args:
        ref_words (list): Reference words.
        hyp_words (list): Hypothesis words.
        
    Returns:
        tuple: (substitutions, deletions, insertions)
    """
    m, n = len(ref_words), len(hyp_words)
    # Initialize the matrix. Each cell holds a tuple: (total_cost, S, D, I)
    dp = [[(0, 0, 0, 0) for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Base cases: empty hypothesis => all deletions; empty reference => all insertions.
    for i in range(1, m + 1):
        dp[i][0] = (i, 0, i, 0)  # i deletions
    for j in range(1, n + 1):
        dp[0][j] = (j, 0, 0, j)  # j insertions

    # Fill dp matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # substitution: cost 1
                sub_cost, sub_S, sub_D, sub_I = dp[i - 1][j - 1]
                sub = (sub_cost + 1, sub_S + 1, sub_D, sub_I)
                # deletion: cost 1
                del_cost, del_S, del_D, del_I = dp[i - 1][j]
                deletion = (del_cost + 1, del_S, del_D + 1, del_I)
                # insertion: cost 1
                ins_cost, ins_S, ins_D, ins_I = dp[i][j - 1]
                insertion = (ins_cost + 1, ins_S, ins_D, ins_I + 1)
                
                # choose the minimum cost option (tie-break arbitrarily)
                dp[i][j] = min(sub, deletion, insertion, key=lambda x: x[0])
                
    # Return errors excluding cost
    total_cost, S, D, I = dp[m][n]
    return S, D, I


def compute_and_print_wer_complements(hypothesis, reference):
    """
    Compute and print 1 - WER for different error types.
    
    Args:
        hypothesis (list): List of hypothesis sequences.
        reference (list): List of reference sequences.
    """
    total_S, total_D, total_I, total_ref_words = 0, 0, 0, 0
    
    # Iterate over each pair of sentences.
    for ref_seq, hyp_seq in zip(reference, hypothesis):
        # Count total words in this reference sentence.
        total_ref_words += len(ref_seq)
        S, D, I = compute_edit_distance(ref_seq, hyp_seq)
        total_S += S
        total_D += D
        total_I += I

    if total_ref_words == 0:
        print("Reference is empty. Cannot compute WER.")
        return

    # Compute WER components as ratios.
    full_wer = (total_S + total_D + total_I) / total_ref_words
    subs_wer = total_S / total_ref_words
    dels_wer = total_D / total_ref_words
    ins_wer = total_I / total_ref_words
    subs_dels_wer = (total_S + total_D) / total_ref_words
    subs_ins_wer = (total_S + total_I) / total_ref_words
    dels_ins_wer = (total_D + total_I) / total_ref_words

    wer_metrics = {
        "Full": full_wer,
        "Substitutions Only": subs_wer,
        "Deletions Only": dels_wer,
        "Insertions Only": ins_wer,
        "Substitutions + Deletions": subs_dels_wer,
        "Substitutions + Insertions": subs_ins_wer,
        "Deletions + Insertions": dels_ins_wer
    }
    
    print("WER Complements (1 - WER):")
    for key, wer in wer_metrics.items():
        print(f"{key}: {1 - wer:.3f}") 