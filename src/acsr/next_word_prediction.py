from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import re
import time
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import wandb
import multiprocessing
import sys
from functools import partial
from nltk.tokenize import word_tokenize, sent_tokenize

# Configuration
CONFIG = { 
    "preprocessed_train_path": "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/preprocessed_train.txt",
    "preprocessed_val_path": "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/preprocessed_val.txt",
    "data_train_path": "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/train.csv",
    "data_val_path": "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/eval.csv",
    "seq_length": 10,
    "batch_size": 3000,
    "embedding_dim": 200,
    "hidden_dim": 512,
    "num_layers": 4,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "epochs": 200,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "min_count": 1,
    "special_tokens": ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
}

# Model Architecture: Next Word Prediction
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

# Dataset Class
class WordDataset(Dataset):
    def __init__(self, sequences, word_to_idx):
        self.sequences = sequences
        self.word_to_idx = word_to_idx
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        input_indices = [self.word_to_idx.get(word, self.word_to_idx["<UNK>"]) 
                         for word in input_seq]
        target_idx = self.word_to_idx.get(target, self.word_to_idx["<UNK>"])
        return torch.tensor(input_indices), torch.tensor(target_idx)

def load_and_clean_data(file_path):
    """Load and clean data from a CSV file."""
    df = pd.read_csv(file_path)
    texts = df['text'].str.lower()
    texts = texts.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    texts = texts.apply(lambda x: re.sub(r'http\S+', 'link', x))
    # Tokenize into sentences using nltk
    texts = texts.apply(sent_tokenize)
    return [sentence for text in texts for sentence in text]

def process_single_sentence(sentence):
    """Process a sentence into a list of word tokens."""
    # Here we use word_tokenize; you could adjust to any tokenizer you prefer.
    tokens = word_tokenize(sentence)
    return tokens

def preprocess_data():
    """Load and preprocess both datasets with caching."""
    train_cached = Path(CONFIG['preprocessed_train_path']).exists()
    val_cached = Path(CONFIG['preprocessed_val_path']).exists()
    
    if train_cached and val_cached:
        print("Loading preprocessed data...")
        sys.stdout.flush()
        return (
            load_preprocessed(CONFIG['preprocessed_train_path']),
            load_preprocessed(CONFIG['preprocessed_val_path'])
        )
    
    print("Preprocessing data...")
    sys.stdout.flush()
    # Process training data
    train_text = load_and_clean_data(CONFIG['data_train_path'])
    with multiprocessing.Pool(20) as pool:
        train_tokens = pool.map(process_single_sentence, train_text)
    
    # Process validation data
    val_text = load_and_clean_data(CONFIG['data_val_path'])
    with multiprocessing.Pool(20) as pool:
        val_tokens = pool.map(process_single_sentence, val_text)
    
    # Filter empty sequences
    train_tokens = [tokens for tokens in train_tokens if tokens]
    val_tokens = [tokens for tokens in val_tokens if tokens]
    
    # Save processed data
    save_preprocessed(train_tokens, CONFIG['preprocessed_train_path'])
    save_preprocessed(val_tokens, CONFIG['preprocessed_val_path'])
    
    return train_tokens, val_tokens

def save_preprocessed(data, path):
    """Save preprocessed data."""
    with open(path, 'w', encoding='utf-8') as f:
        for tokens in data:
            f.write(' '.join(tokens) + '\n')

def load_preprocessed(path):
    """Load preprocessed data."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().split(' ') for line in f]
    
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def build_vocab(train_tokens):
    """Build vocabulary from training data only."""
    # Start with special tokens.
    vocab = CONFIG['special_tokens'].copy()
    # Gather all words from training data.
    valid_words = [word for tokens in train_tokens for word in tokens]
    valid_words = unique(valid_words)
    vocab += valid_words
    vocab = unique(vocab)
    
    # Optionally, you can filter out words based on frequency here.
    return {word: idx for idx, word in enumerate(vocab)}, vocab

def create_datasets(train_tokens, val_tokens, word_to_idx):
    """Create Dataset objects for both splits."""
    def process_split(tokens_list, train=True):
        sequences = []
        for tokens in tokens_list:
            # Add start and end tokens
            tokens = ['<SOS>'] + tokens + ['<EOS>']
            step = 10 if train else 1
            for i in range(0, len(tokens) - 1, 1):
                # Use a window of seq_length tokens (pad if necessary)
                start_idx = max(0, i - CONFIG['seq_length'] + 1)
                seq = tokens[start_idx:i+1]
                padded = ['<PAD>'] * (CONFIG['seq_length'] - len(seq)) + seq
                # Optionally, filter out sequences that are too short
                non_special = [t for t in padded if t not in ['<PAD>', '<SOS>']]
                if len(non_special) < 1:
                    continue
                sequences.append((padded, tokens[i+1]))
        return sequences

    return (
        WordDataset(process_split(train_tokens), word_to_idx),
        WordDataset(process_split(val_tokens, train=False), word_to_idx)
    )

# Modified training pipeline
def train_model(train_dataset, val_dataset, model):
    wandb.init(project="word-prediction", config=CONFIG)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=10, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # Training loop
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
            torch.save(model.state_dict(), "/pasteur/appa/homes/bsow/ACSR/src/acsr/saved_models/word_model_last.pth")
            wandb.save("word_model_last.pth")
            
if __name__ == "__main__":
    # Load and preprocess data
    train_tokens, val_tokens = preprocess_data()
    
    # Build vocabulary from training data
    word_to_idx, vocab = build_vocab(train_tokens)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    CONFIG["vocab_size"] = len(vocab)
    print(f"Vocabulary size: {len(vocab)}")
    sys.stdout.flush()
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(train_tokens, val_tokens, word_to_idx)
    print("Len train dataset: ", len(train_dataset))
    print("Len val dataset: ", len(val_dataset))
    
    # Print some samples from the training dataset
    print("Train dataset samples:")
    for i in range(10):
        print([idx_to_word[idx] for idx in train_dataset[i][0].numpy()])
    
    # Initialize model
    model = NextWordLSTM(
        vocab_size=len(vocab),
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    # Start training
    wandb.login(key="580ab03d7111ed25410f9831b06b544b5f5178a2")
    train_model(train_dataset, val_dataset, model)
    
    model.eval()
    
    # Select 10 random validation samples for analysis
    import numpy as np
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_dataset), 10, replace=False)
    
    print("\nModel Predictions Analysis:")
    for i, idx in enumerate(sample_indices):
        input_seq, target = val_dataset[idx]
        input_words = [idx_to_word[i.item()] for i in input_seq]
        true_next = idx_to_word[target.item()]
        input_tensor = input_seq.unsqueeze(0).to(CONFIG['device'])
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, top_preds = torch.topk(probabilities, k=3)
        top_words = [idx_to_word[idx.item()] for idx in top_preds[0]]
        context = ' '.join(input_words).replace('<PAD>', '_').replace('<SOS>', '[START]')
        
        print(f"\nSample {i+1}:")
        print(f"Input Context: {context}")
        print(f"True Next Word: {true_next}")
        print(f"Top 3 Predictions: {', '.join(top_words)}")
        print(f"Probability Distribution: {probabilities[0].cpu().numpy().round(3)}")
