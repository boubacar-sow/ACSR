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
from decoder_lm import text_to_ipa, syllabify_ipa, convert_ipa_to_syllables

# Configuration
CONFIG = { 
    "preprocessed_train_path": "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/preprocessed_train.txt",
    "preprocessed_val_path": "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/preprocessed_val.txt",
    "data_train_path": "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/concat_train.csv",
    "data_val_path": "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/eval.csv",
    "preprocessed_path": "/pasteur/appa/homes/bsow/ACSR/data/news_dataset/preprocessed_syllables.txt",
    "seq_length": 15,
    "batch_size": 2048,
    "embedding_dim": 200,
    "hidden_dim": 512,
    "num_layers": 4,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "epochs": 200,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vocab_size": 329,
    "min_count": 1,
    "special_tokens": ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
}

# Phoneme mapping 
IPA_TO_TARGET = {
    # Vowels
    "a": "a", "ɑ": "a", "ə": "x", "ɛ": "e^", "ø": "x", "œ": "x^", "i": "i", "y": "y", "e": "e",
    "u": "u", "ɔ": "o", "o": "o^", "ɑ̃": "a~", "ɛ̃": "e~", "ɔ̃": "o~", "œ̃": "x~",
    " ": " ",  # Space

    # Consonants
    "b": "b", "c": "k", "d": "d", "f": "f", "ɡ": "g", "j": "j", "k": "k", "l": "l", 
    "m": "m", "n": "n", "p": "p", "s": "s", "t": "t", "v": "v", "w": "w", "z": "z", 
    "ɥ": "h", "ʁ": "r", "ʃ": "s^", "ʒ": "z^", "ɲ": "gn", "ŋ": "ng"
}


# Model Architecture
class NextSyllableLSTM(nn.Module):
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
class SyllableDataset(Dataset):
    def __init__(self, sequences, syllable_to_idx):
        self.sequences = sequences
        self.syllable_to_idx = syllable_to_idx
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        input_indices = [self.syllable_to_idx.get(syl, self.syllable_to_idx["<UNK>"]) 
                        for syl in input_seq]
        target_idx = self.syllable_to_idx.get(target, self.syllable_to_idx["<UNK>"])
        return torch.tensor(input_indices), torch.tensor(target_idx)

def load_and_clean_data(file_path):
    """Load and clean data from a specific file"""
    df = pd.read_csv(file_path)
    texts = df['text'].str.lower()
    #texts = texts.apply(lambda x: re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', x))
    texts = texts.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    texts = texts.apply(lambda x: re.sub(r'http\S+', 'link', x))
    texts = texts.apply(lambda x: re.sub(r'iel', 'il', x))
    from nltk.tokenize import sent_tokenize
    texts = texts.apply(sent_tokenize)
    return [sentence for text in texts for sentence in text]

def process_single_sentence(sentence, IPA_TO_TARGET):
    """Process a single sentence into IPA syllables"""
    ipa = text_to_ipa(sentence)
    new_syllables = convert_ipa_to_syllables(ipa, IPA_TO_TARGET)
    if "<UNK>" in new_syllables:
        return []
    return new_syllables

def preprocess_data():
    """Load and preprocess both datasets with separate caching"""
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

def save_preprocessed(data, path):
    """Save preprocessed data with proper formatting"""
    with open(path, 'w', encoding='utf-8') as f:
        for syllables in data:
            f.write(''.join(syllables) + '\n')

def load_preprocessed(path):
    """Load preprocessed data"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.replace('\n','').split(' ') for line in f]
    
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def build_vocab(train_syllables):
    """Build vocabulary from training data only"""
    counter = Counter()
    import csv
    # Filter and sort vocabulary
    #vocab = CONFIG['special_tokens'].copy()
    # load all the syllables from /pasteur/appa/homes/bsow/ACSR/data/training_videos/syllables.txt
    with open(r"/pasteur/appa/homes/bsow/ACSR/data/french_dataset/vocab.txt", "r") as file:
        reader = csv.reader(file)
        vocabulary_list = [row[0] for row in reader]
    vocab = unique(vocabulary_list)
    #vocab = unique(vocab)
    valid_syllables = [syl for sequence in train_syllables for syl in sequence]
    valid_syllables = unique(valid_syllables)
    vocab += valid_syllables
    vocab = unique(vocab)
    # save the vocab to /pasteur/appa/homes/bsow/ACSR/data/news_dataset/vocab.txt
    with open("/pasteur/appa/homes/bsow/ACSR/data/french_dataset/vocab2.txt", "w") as f:
        for syl in vocab:
            f.write(syl + "\n")
    
    return {syl: idx for idx, syl in enumerate(vocab)}, vocab

def create_datasets(train_syllables, val_syllables, syllable_to_idx):
    """Create Dataset objects for both splits"""
    def process_split(syllables):
        sequences = []
        for syll_list in syllables:
            syll_list = [s.replace(' ', '').replace("\n", "") for s in syll_list]
            tokens = ['<SOS>'] + syll_list + ['<EOS>']
            for i in range(0, len(tokens) - 2):
                start_idx = max(0, i - CONFIG['seq_length'] + 1)
                seq = tokens[start_idx:i+1]
                padded = ['<PAD>'] * (CONFIG['seq_length'] - len(seq)) + seq
                non_special = [t for t in padded if t not in ['<PAD>', '<SOS>']]
                if len(non_special) < 5:
                    continue
                sequences.append((padded, tokens[i+1]))
        return sequences

    return (
        SyllableDataset(process_split(train_syllables), syllable_to_idx),
        SyllableDataset(process_split(val_syllables), syllable_to_idx)
    )


# Modified training pipeline
def train_model(train_dataset, val_dataset, model):
    wandb.init(project="ipa-syllable-prediction", config=CONFIG)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=10, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # Training loop
        start = time.time()
        model.train()
        train_loss = 0.0  # Total loss (sum of batch losses)
        correct_train = 0
        correct_top10_train = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Multiply loss by batch size (to account for variable batch sizes)
            batch_loss = loss.item() * inputs.size(0)
            train_loss += batch_loss

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == targets).sum().item()

            _, predicted_top10 = torch.topk(outputs, k=10, dim=1)
            correct_top10_train += torch.sum(predicted_top10 == targets.view(-1, 1)).item()
        
        # Validation loop
        model.eval()
        val_loss = 0.0  # Total validation loss
        correct = 0
        correct_top5 = 0
        correct_top10 = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
                outputs = model(inputs)
                # Multiply loss by batch size
                batch_loss = criterion(outputs, targets).item() * inputs.size(0)
                val_loss += batch_loss
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()

                _, predicted_top5 = torch.topk(outputs, k=5, dim=1)
                correct_top5 += torch.sum(predicted_top5 == targets.view(-1, 1)).item()

                _, predicted_top10 = torch.topk(outputs, k=10, dim=1)
                correct_top10 += torch.sum(predicted_top10 == targets.view(-1, 1)).item()
        
        # Calculate metrics (divide by dataset length)
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        train_acc = correct_train / len(train_dataset)
        train_top10_acc = correct_top10_train / len(train_dataset)
        val_acc = correct / len(val_dataset)
        val_top5_acc = correct_top5 / len(val_dataset)
        val_top10_acc = correct_top10 / len(val_dataset)
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        end = time.time()
        
        # Print and log metrics
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
        
        # Save best model using correct val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "syllable_model_last.pth")
            wandb.save("syllable_model_last.pth")
            
if __name__ == "__main__":
    # Load and preprocess data
    train_syllables, val_syllables = preprocess_data()
    
    # Build vocabulary from training data
    syllable_to_idx, vocab = build_vocab(train_syllables)
    # Create reverse vocabulary mapping
    idx_to_syllable = {v: k for k, v in syllable_to_idx.items()}
#
    CONFIG["vocab_size"] = len(vocab)
    print(f"Vocabulary size: {len(vocab)}")
    print(syllable_to_idx)
    sys.stdout.flush()
    # Create datasets
    train_dataset, val_dataset = create_datasets(train_syllables, val_syllables, syllable_to_idx)
    print("Len train dataset: ", len(train_dataset))
    print("Len val dataset: ", len(val_dataset))
    # print samples from the dataset
    print("Train dataset samples:")
    for i in range(10):
        print([idx_to_syllable[idx] for idx in train_dataset[i][0].numpy()])
    print("\nVal dataset samples:")
    for i in range(10):
        print([idx_to_syllable[idx] for idx in val_dataset[i][0].numpy()])
    sys.stdout.flush()
##
    # Initialize model
    model = NextSyllableLSTM(
        vocab_size=len(vocab),
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    # Start training
    wandb.login(key="580ab03d7111ed25410f9831b06b544b5f5178a2")
    # load model
    #model.load_state_dict(torch.load("best_syllable_model_def2.pth"))
#
    train_model(train_dataset, val_dataset, model)
##
    model.eval()
    
    # Select 5 random validation samples
    import numpy as np
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_dataset), 10, replace=False)
    
    print("\nModel Predictions Analysis:")
    for i, idx in enumerate(sample_indices):
        # Get raw sequence and target
        input_seq, target = val_dataset[idx]
        
        # Convert indices to syllables
        input_syllables = [idx_to_syllable[i.item()] for i in input_seq]
        true_next = idx_to_syllable[target.item()]
        
        # Format input for model
        input_tensor = input_seq.unsqueeze(0).to(CONFIG['device'])  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, top_preds = torch.topk(probabilities, k=3)
            
        # Convert predictions to syllables
        top_syllables = [idx_to_syllable[idx.item()] for idx in top_preds[0]]
        
        # Clean up display
        context = ' '.join(input_syllables).replace('<PAD>', '_').replace('<SOS>', '[START]')
        
        print(f"\nSample {i+1}:")
        print(f"Input Context: {context}")
        print(f"True Next Syllable: {true_next}")
        print(f"Top 3 Predictions: {', '.join(top_syllables)}")
        print(f"Probability Distribution: {probabilities[0].cpu().numpy().round(3)}")