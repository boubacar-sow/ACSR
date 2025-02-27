from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import re
import time
import subprocess
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import wandb
import multiprocessing
import sys
from functools import partial
from decoder_lm import clean_text, text_to_ipa, syllabify_ipa, convert_ipa_to_syllables

# Configuration (added "context_window" for CBOW)
CONFIG = { 
    "preprocessed_train_path": "/scratch2/bsow/Documents/ACSR/data/french_dataset/preprocessed_train.txt",
    "preprocessed_val_path": "/scratch2/bsow/Documents/ACSR/data/french_dataset/preprocessed_val.txt",
    "data_train_path": "/scratch2/bsow/Documents/ACSR/data/french_dataset/concat_train.csv",
    "data_val_path": "/scratch2/bsow/Documents/ACSR/data/french_dataset/preprocessed_val.txt",
    "preprocessed_path": "/scratch2/bsow/Documents/ACSR/data/news_dataset/preprocessed_syllables.txt",
    "batch_size": 512,
    "embedding_dim": 200,
    "hidden_dim": 256,  # If you want to add a hidden layer later, you can use this
    "num_layers": 4,    # Not used for CBOW; kept for consistency with your config
    "dropout": 0.2,
    "learning_rate": 0.001,
    "epochs": 200,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vocab_size": 270,  # Will be updated after building the vocab
    "min_count": 1,
    "special_tokens": ["<PAD>", "<UNK>", "<SOS>", "<EOS>"],
    "context_window": 7  # Number of tokens to consider on each side for the CBOW model
}

# Phoneme mapping 
IPA_TO_TARGET = {
    # Vowels
    "a": "a", "ɑ": "a", "ə": "x", "ɛ": "e^", "ø": "x", "œ": "e^", "i": "i", "y": "y", "e": "e",
    "u": "u", "o": "o", "ɔ": "o^", "ɑ̃": "a~", "ɛ̃": "e~", "ɔ̃": "o~", "œ̃": "x~",
    " ": " ",  # Space

    # Consonants
    "b": "b", "c": "k", "d": "d", "f": "f", "ɡ": "g", "j": "j", "k": "k", "l": "l", 
    "m": "m", "n": "n", "p": "p", "s": "s", "t": "t", "v": "v", "w": "w", "z": "z", 
    "ɥ": "h", "ʁ": "r", "ʃ": "s^", "ʒ": "z^", "ɲ": "gn", 
}

# ------------------------------
# CBOW Model Architecture
# ------------------------------
import torch
import torch.nn as nn

class AttentionCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Linear layer to compute attention scores for each context token
        self.attn = nn.Linear(embedding_dim, 1)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context):
        # context: [batch_size, context_length]
        embedded = self.embedding(context)               # [batch_size, context_length, embedding_dim]
        # Compute attention scores and weights
        attn_scores = self.attn(embedded).squeeze(-1)      # [batch_size, context_length]
        attn_weights = torch.softmax(attn_scores, dim=1)   # [batch_size, context_length]
        # Compute weighted sum of embeddings
        context_vector = torch.sum(embedded * attn_weights.unsqueeze(-1), dim=1)
        x = self.dropout(context_vector)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output


# ------------------------------
# Dataset Class (remains largely the same)
# ------------------------------
class SyllableDataset(Dataset):
    def __init__(self, sequences, syllable_to_idx):
        self.sequences = sequences
        self.syllable_to_idx = syllable_to_idx
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        context, target = self.sequences[idx]
        # Convert the context syllables to indices
        context_indices = [self.syllable_to_idx.get(syl, self.syllable_to_idx["<UNK>"]) 
                           for syl in context]
        target_idx = self.syllable_to_idx.get(target, self.syllable_to_idx["<UNK>"])
        return torch.tensor(context_indices), torch.tensor(target_idx)

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
def load_and_clean_data(file_path):
    """Load and clean data from a specific file"""
    df = pd.read_csv(file_path)
    texts = df['text'].str.lower()
    texts = texts.apply(lambda x: re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', x))
    texts = texts.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    texts = texts.apply(lambda x: re.sub(r'http\S+', '', x))
    from nltk.tokenize import sent_tokenize
    texts = texts.apply(sent_tokenize)
    return [sentence for text in texts for sentence in text]

def process_single_sentence(sentence, IPA_TO_TARGET):
    """Process a single sentence into IPA syllables"""
    ipa = text_to_ipa(sentence)
    syllables = syllabify_ipa(ipa)
    new_syllables = convert_ipa_to_syllables(" ".join(syllables), IPA_TO_TARGET)
    if "<UNK>" in new_syllables:
        return []
    return new_syllables

def save_preprocessed(data, path):
    """Save preprocessed data with proper formatting"""
    with open(path, 'w', encoding='utf-8') as f:
        for syllables in data:
            f.write(''.join(syllables) + '\n')

def load_preprocessed(path):
    """Load preprocessed data"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.split('   ') for line in f]

def preprocess_data():
    """Load and preprocess both datasets with separate caching"""
    train_cached = Path(CONFIG['preprocessed_train_path']).exists()
    val_cached = Path(CONFIG['preprocessed_val_path']).exists()
    
    if train_cached and val_cached:
        print("Loading preprocessed data...")
        return (
            load_preprocessed(CONFIG['preprocessed_train_path']),
            load_preprocessed(CONFIG['preprocessed_val_path'])
        )
    
    print("Preprocessing data...")
    # Process training data
    train_text = load_and_clean_data(CONFIG['data_train_path'])
    with multiprocessing.Pool(20) as pool:
        train_syllables = pool.map(partial(process_single_sentence, IPA_TO_TARGET=IPA_TO_TARGET), train_text)
    
    # Process validation data
    val_text = load_and_clean_data(CONFIG['data_val_path'])
    with multiprocessing.Pool(20) as pool:
        val_syllables = pool.map(partial(process_single_sentence, IPA_TO_TARGET=IPA_TO_TARGET), val_text)
    
    # Filter empty sequences
    train_syllables = [s for s in train_syllables if s]
    val_syllables = [s for s in val_syllables if s]
    
    # Save processed data
    save_preprocessed(train_syllables, CONFIG['preprocessed_train_path'])
    save_preprocessed(val_syllables, CONFIG['preprocessed_val_path'])
    
    return train_syllables, val_syllables

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def build_vocab(train_syllables):
    """Build vocabulary from training data only"""
    counter = Counter()
    for syllables in train_syllables:
        counter.update(syllables)

    # Start with special tokens
    vocab = CONFIG['special_tokens'].copy()
    # Load all the syllables from your vocab file
    with open("/scratch2/bsow/Documents/ACSR/data/news_dataset/vocab.txt", "r") as f:
        syllables = f.readlines()
        syllables = [syl.replace(" ", "").replace("\n", "") for syl in syllables]
    vocab += syllables
    
    vocab = unique(vocab)
    return {syl: idx for idx, syl in enumerate(vocab)}, vocab

# ------------------------------
# Create CBOW Datasets
# ------------------------------
def create_cbow_datasets(train_syllables, val_syllables, syllable_to_idx):
    """
    For each sentence, generate CBOW training pairs.
    For each target word (excluding <SOS> and <EOS>), the context is taken from
    a window of size CONFIG['context_window'] tokens on each side.
    If the context is smaller than 2*context_window, we pad with <PAD>.
    """
    def process_split(syllables):
        sequences = []
        window = CONFIG['context_window']
        for syll_list in syllables:
            # Clean up syllables
            syll_list = [s.replace("\n", "") for s in syll_list][0].split(" ")
            # Add special tokens at beginning and end
            tokens = ['<SOS>'] + syll_list + ['<EOS>']
            # For each token position (skip the very first and last tokens)
            for i in range(1, len(tokens)-1):
                # Get left context (up to "window" tokens)
                left_context = tokens[max(0, i - window): i]
                # Get right context (up to "window" tokens)
                right_context = tokens[i+1: i+window+1]
                context = left_context + right_context
                # Pad context if needed to get fixed length (2 * window)
                while len(context) < 2 * window:
                    context.append('<PAD>')
                sequences.append((context, tokens[i]))
        return sequences
    
    return (
        SyllableDataset(process_split(train_syllables), syllable_to_idx),
        SyllableDataset(process_split(val_syllables), syllable_to_idx)
    )

# ------------------------------
# Training Pipeline
# ------------------------------
def train_model(train_dataset, val_dataset, model):
    wandb.init(project="ipa-syllable-cbow", config=CONFIG)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        start = time.time()
        model.train()
        train_loss = 0.0
        correct_train = 0
        correct_top10_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item() * inputs.size(0)
            train_loss += batch_loss

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == targets).sum().item()

            _, predicted_top10 = torch.topk(outputs, k=10, dim=1)
            correct_top10_train += torch.sum(predicted_top10 == targets.view(-1, 1)).item()
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        correct_top5 = 0
        correct_top10 = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets).item() * inputs.size(0)
                val_loss += batch_loss
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()

                _, predicted_top5 = torch.topk(outputs, k=5, dim=1)
                correct_top5 += torch.sum(predicted_top5 == targets.view(-1, 1)).item()

                _, predicted_top10 = torch.topk(outputs, k=10, dim=1)
                correct_top10 += torch.sum(predicted_top10 == targets.view(-1, 1)).item()
        
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        train_acc = correct_train / len(train_dataset)
        train_top10_acc = correct_top10_train / len(train_dataset)
        val_acc = correct / len(val_dataset)
        val_top5_acc = correct_top5 / len(val_dataset)
        val_top10_acc = correct_top10 / len(val_dataset)
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        end = time.time()
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Train Top-10 Acc: {train_top10_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val Top-5 Acc: {val_top5_acc:.4f} | "
              f"Val Top-10 Acc: {val_top10_acc:.4f} | "
              f"Perplexity: {perplexity:.2f} | "
              f"Time: {end-start:.2f}s")
        
        sys.stdout.flush()
        wandb.log({
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "train_top10_acc": train_top10_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_top5_acc": val_top5_acc,
            "val_top10_acc": val_top10_acc,
            "perplexity": perplexity
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_syllable_cbow_model.pth")
            wandb.save("best_syllable_cbow_model.pth")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # Load and preprocess data
    train_syllables, val_syllables = preprocess_data()
    
    # Build vocabulary from training data
    syllable_to_idx, vocab = build_vocab(train_syllables)
    idx_to_syllable = {v: k for k, v in syllable_to_idx.items()}

    CONFIG["vocab_size"] = len(vocab)
    print(f"Vocabulary size: {len(vocab)}")
    print(syllable_to_idx)
 
    # Create CBOW datasets
    train_dataset, val_dataset = create_cbow_datasets(train_syllables, val_syllables, syllable_to_idx)

    # Print a few samples from the dataset
    print("Train dataset samples (context -> target):")
    for i in range(10):
        context_idxs, target_idx = train_dataset[i]
        context_tokens = [idx_to_syllable[idx.item()] for idx in context_idxs.numpy()]
        target_token = idx_to_syllable[target_idx.item()]
        print(f"Context: {context_tokens}  --> Target: {target_token}")
    print("\nVal dataset samples (context -> target):")
    for i in range(10):
        context_idxs, target_idx = val_dataset[i]
        context_tokens = [idx_to_syllable[idx.item()] for idx in context_idxs.numpy()]
        target_token = idx_to_syllable[target_idx.item()]
        print(f"Context: {context_tokens}  --> Target: {target_token}")
    sys.stdout.flush()
    # Initialize the CBOW model
    model = AttentionCBOW(
        vocab_size=len(vocab),
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    # Login to wandb (ensure your API key is set up securely)
    wandb.login(key="580ab03d7111ed25410f9831b06b544b5f5178a2")
    train_model(train_dataset, val_dataset, model)
    
    model.eval()
    
    # Model Predictions Analysis
    import numpy as np
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_dataset), 10, replace=False)
    
    print("\nModel Predictions Analysis:")
    for i, idx in enumerate(sample_indices):
        input_context, target = val_dataset[idx]
        input_tokens = [idx_to_syllable[i.item()] for i in input_context]
        true_target = idx_to_syllable[target.item()]
        
        # Prepare input for model (adding batch dimension)
        input_tensor = input_context.unsqueeze(0).to(CONFIG['device'])
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, top_preds = torch.topk(probabilities, k=3)
            
        top_tokens = [idx_to_syllable[idx.item()] for idx in top_preds[0]]
        context_display = ' '.join(input_tokens).replace('<PAD>', '_').replace('<SOS>', '[START]')
        
        print(f"\nSample {i+1}:")
        print(f"Input Context: {context_display}")
        print(f"True Target Syllable: {true_target}")
        print(f"Top 3 Predictions: {', '.join(top_tokens)}")
        print(f"Probability Distribution: {probabilities[0].cpu().numpy().round(3)}")
