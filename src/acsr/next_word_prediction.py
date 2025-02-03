import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import numpy as np
import os
import wandb
import time
from collections import Counter
from nltk.tokenize import word_tokenize
import sys

# Configuration
CONFIG = {
    "data_path": "/scratch2/bsow/Documents/ACSR/data/news_dataset/summarization_validation.csv",
    "seq_length": 12,  # Number of words in input sequence
    "batch_size": 512,
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 3,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "epochs": 10,
    "vocab_size": 32000,  # Limit vocabulary size
    "min_word_freq": 2,  # Minimum word frequency to keep in vocabulary
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Step 1: Data Preprocessing
class FrenchNewsDataset(Dataset):
    def __init__(self, texts, word_to_idx, seq_length):
        self.word_to_idx = word_to_idx
        self.seq_length = seq_length
        self.sequences = self._create_sequences(texts)
        
    def _create_sequences(self, texts):
        sequences = []
        for text in texts:
            tokens = ['<SOS>'] + word_tokenize(text) + ['<EOS>']
            
            # Create sequences for all possible positions
            for i in range(len(tokens) - 1):  # -1 to ensure target exists
                # Get input window (minimum 2 tokens)
                start_idx = max(0, i - self.seq_length + 1)
                input_seq = tokens[start_idx:i+1]
                # Pad sequence to seq_length (left-padding)
                pad_length = self.seq_length - len(input_seq)
                padded_seq = ['<PAD>'] * pad_length + input_seq
                target = tokens[i+1]      
                sequences.append((padded_seq, target))
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        seq_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                      for word in seq]
        target_idx = self.word_to_idx.get(target, self.word_to_idx['<UNK>'])
        return torch.tensor(seq_indices), torch.tensor(target_idx)

# Step 2: Model Architecture
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

# Step 3: Training Pipeline
def train_model(dataset, model):
    wandb.init(project="french-news-lm", config=CONFIG)
    
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'], shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        correct_train = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        correct_top5 = 0
        correct_top10 = 0
        correct_top20 = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                # Calculate top-5 accuracy
                _, predicted_top5 = torch.topk(outputs, k=5, dim=1)
                correct_top5 += torch.sum(predicted_top5 == targets.view(-1, 1)).item()

                # Calculate top-10 accuracy
                _, predicted_top10 = torch.topk(outputs, k=10, dim=1)
                correct_top10 += torch.sum(predicted_top10 == targets.view(-1, 1)).item()

                # Calculate top-20 accuracy
                _, predicted_top20 = torch.topk(outputs, k=20, dim=1)
                correct_top20 += torch.sum(predicted_top20 == targets.view(-1, 1)).item()
        
        # Metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = correct_train / len(train_data)
        val_acc = correct / total
        val_acc_top5 = correct_top5 / total
        val_acc_top10 = correct_top10 / total
        val_acc_top20 = correct_top20 / total
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        # Reporting
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Perplexity: {perplexity:.2f} | "
              f"Val Acc Top-5: {val_acc_top5:.4f} | "
            f"Val Acc Top-10: {val_acc_top10:.4f} | "
            f"Val Acc Top-20: {val_acc_top20:.4f} | "
              f"Time: {time.time()-start_time:.2f}s")
        sys.stdout.flush()
        # Logging
        wandb.log({
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_acc_top5": val_acc_top5,
            "val_acc_top10": val_acc_top10,
            "val_acc_top20": val_acc_top20,
            "perplexity": perplexity
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            torch.save(optimizer.state_dict(), "best_optimizer.pth")
            wandb.save("best_model.pth")
    
    wandb.finish()

# Helper functions
def load_and_clean_data():
    df = pd.read_csv(CONFIG['data_path'])
    texts = df['text'].str.lower()
    texts = texts.apply(lambda x: re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', '', x))
    texts = texts.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return texts.tolist()

def build_vocab(texts):
    word_counts = Counter()
    for text in texts:
        word_counts.update(word_tokenize(text))
    
    # Filter rare words
    vocab = [word for word, count in word_counts.items() if count >= CONFIG['min_word_freq']]
    vocab = vocab[:CONFIG['vocab_size']-3]  # Reserve spots for special tokens
    
    # Add special tokens
    vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + vocab
    return {word: idx for idx, word in enumerate(vocab)}, vocab

# Main execution
if __name__ == "__main__":
    # wandb login
    wandb.login(key="580ab03d7111ed25410f9831b06b544b5f5178a2")

    # Load and process data
    texts = load_and_clean_data()
    word_to_idx, vocab = build_vocab(texts)
    print(f"Vocabulary size: {len(vocab)}")
    sys.stdout.flush()
    
    # Create dataset
    full_dataset = FrenchNewsDataset(
        texts=texts,
        word_to_idx=word_to_idx,
        seq_length=CONFIG['seq_length']
    )
    
    # Initialize model
    model = NextWordLSTM(
        vocab_size=len(vocab),
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    # Start training
    train_model(full_dataset, model)

    # After training
    model.eval()  # Always set to evaluation mode before testing

    # Test the model on a sample input
    test_sentence = "je suis très content"
    test_words = test_sentence.split()

    # Convert to indices with proper sequence formatting
    seq_length = CONFIG['seq_length']

    # Add SOS token and pad to seq_length
    padded_sequence = ['<SOS>'] + test_words
    padded_sequence = padded_sequence[-seq_length:]  # Take last seq_length tokens
    if len(padded_sequence) < seq_length:
        # Left-pad with PAD tokens
        padded_sequence = ['<PAD>'] * (seq_length - len(padded_sequence)) + padded_sequence

    # Convert to indices
    test_indices = [
        word_to_idx.get(word, word_to_idx['<UNK>'])
        for word in padded_sequence
    ]

    # Initialize prediction loop
    predicted_sentence = test_sentence
    predicted_words = test_words.copy()
    max_tokens = 10  # Maximum number of tokens to predict

    # Predict next tokens
    with torch.no_grad():
        for _ in range(max_tokens):
            # Create input tensor
            input_tensor = torch.tensor(test_indices).unsqueeze(0).to(CONFIG['device'])
            
            # Get model output
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_word = vocab[predicted_idx.item()]
            
            # Stop if EOS is predicted
            if predicted_word == '<EOS>':
                break
            
            # Update predicted sentence and sequence
            predicted_sentence += " " + predicted_word
            predicted_words.append(predicted_word)
            
            # Update input sequence for next prediction
            test_indices = test_indices[1:] + [predicted_idx.item()]  # Slide window

    # Format output
    print(f"\nTest Case:")
    print(f"Input: {test_sentence}")
    print(f"Predicted Continuation: {predicted_sentence}")
    print(f"Predicted Words: {predicted_words}")
