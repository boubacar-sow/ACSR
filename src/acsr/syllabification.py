import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import sys
import os
import wandb
import time

# Step 1: Preprocess the Data
class SyllableDataset(Dataset):
    def __init__(self, syllabized_ipa_sentences, seq_length=5):
        self.syllabized_ipa_sentences = syllabized_ipa_sentences
        self.seq_length = seq_length
        self.syllables = self._get_syllables()
        self.syllable_to_idx = {syllable: i for i, syllable in enumerate(list(set(self.syllables)))}
        # add space
        self.syllable_to_idx[' '] = len(self.syllable_to_idx)
        self.idx_to_syllable = {i: syllable for syllable, i in self.syllable_to_idx.items()}
        self.vocab_size = len(self.syllable_to_idx)
        print(f"Number of unique syllables: {self.vocab_size}")
        self.data = self._create_sequences()
        print(f"Number of sequences: {len(self.data)}")
    def _get_syllables(self):
        with open("/scratch2/bsow/Documents/ACSR/data/training_videos/syllables.txt", "r", encoding="utf-8") as file:
            syllables = file.read().splitlines()
        return syllables

    def _create_sequences(self):
        sequences = []
        for sentence in self.syllabized_ipa_sentences:
            for i in range(len(sentence) - self.seq_length):
                input_seq = sentence[i:i + self.seq_length]
                output_seq = sentence[i + self.seq_length]
                sequences.append((input_seq, output_seq))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, output_seq = self.data[idx]
        input_indices = [self.syllable_to_idx.get(syllable, self.syllable_to_idx['<UNK>']) for syllable in input_seq]
        output_index = self.syllable_to_idx.get(output_seq, self.syllable_to_idx['<UNK>'])
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(output_index, dtype=torch.long)

# Step 2: Build the Model
class NextSyllableLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(NextSyllableLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

def train_model(dataset, model, epochs=10, batch_size=32, learning_rate=0.001, save_dir="saved_models", device="cuda"):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

        torch.cuda.empty_cache()
        
        model.eval()
        val_loss = 0
        correct_top1 = 0
        correct_top5 = 0
        correct_top10 = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

                # Calculate top-1 accuracy
                _, predicted_top1 = torch.max(outputs, 1)
                correct_top1 += (predicted_top1 == targets).sum().item()

                # Calculate top-5 accuracy
                _, predicted_top5 = torch.topk(outputs, k=5, dim=1)
                correct_top5 += torch.sum(predicted_top5 == targets.view(-1, 1)).item()

                # Calculate top-10 accuracy
                _, predicted_top10 = torch.topk(outputs, k=10, dim=1)
                correct_top10 += torch.sum(predicted_top10 == targets.view(-1, 1)).item()

                total += targets.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy_top1 = correct_top1 / total
        val_accuracy_top5 = correct_top5 / total
        val_accuracy_top10 = correct_top10 / total
        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()  # Calculate perplexity
        end = time.time()
        
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train Loss: {avg_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}, "
            f"Validation Perplexity: {val_perplexity:.4f}, "  # Added perplexity
            f"Validation Top-1 Accuracy: {val_accuracy_top1:.4f}, "
            f"Validation Top-5 Accuracy: {val_accuracy_top5:.4f}, "
            f"Validation Top-10 Accuracy: {val_accuracy_top10:.4f}, "
            f"Time taken: {end - start:.2f} sec"
        )
        sys.stdout.flush()

        # Log metrics to wandb
        wandb.log({
            "val_loss": avg_val_loss,
            "val_perplexity": val_perplexity,  # Log perplexity
            "val_accuracy_top1": val_accuracy_top1,
            "val_accuracy_top5": val_accuracy_top5,
            "val_accuracy_top10": val_accuracy_top10,
            "epoch": epoch + 1
        })

        # Save the model if validation top-1 accuracy improves
        if val_accuracy_top1 > best_val_accuracy:
            best_val_accuracy = val_accuracy_top1
            model_save_path = os.path.join(save_dir, f"best_model_epoch_{epoch + 1}_accuracy_{val_accuracy_top1:.4f}.pth")
            torch.save(model.state_dict(), model_save_path)
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "best_optimizer.pth"))
            print(f"Model saved to {model_save_path} with validation top-1 accuracy: {val_accuracy_top1:.4f}")
            sys.stdout.flush()
            wandb.save(model_save_path)

            
# Step 4: Main Script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sys.stdout.flush()

    syllabized_ipa_sentences = []
    with open("/scratch2/bsow/Documents/ACSR/data/claire_dialogue/syllabized_ipa_train.txt", "r", encoding="utf-8") as file:
        for line in file:
            syllabized_ipa_sentences.append(line.strip().split())

    epochs, batch_size, hidden_dim, embedding_dim, num_layers, seq_length, learning_rate = 300, 256, 256, 128, 4, 10, 0.001
    dataset = SyllableDataset(syllabized_ipa_sentences[:100000], seq_length=seq_length)
    model = NextSyllableLSTM(vocab_size=dataset.vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    # Initialize wandb
    wandb.login(key="580ab03d7111ed25410f9831b06b544b5f5178a2")
    wandb.init(project="next-syllable-lstm", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "seq_length": seq_length,
        "vocab_size": dataset.vocab_size,
        "learning_rate": learning_rate,
        "device": device.type
    })

    train_model(dataset, model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, device=device)

    final_model_path = "next_syllable_lstm.pth"
    torch.save(model.state_dict(), final_model_path)
    print("Model saved to next_syllable_lstm.pth")
    wandb.save(final_model_path)

    wandb.finish()