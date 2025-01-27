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

# Step 1: Preprocess the Data
class SyllableDataset(Dataset):
    def __init__(self, syllabized_ipa_sentences, seq_length=5):
        self.syllabized_ipa_sentences = syllabized_ipa_sentences
        self.seq_length = seq_length
        self.syllables = self._get_syllables()
        self.syllable_to_idx = {syllable: i for i, syllable in enumerate(list(set(self.syllables)))}
        self.idx_to_syllable = {i: syllable for syllable, i in self.syllable_to_idx.items()}
        self.vocab_size = len(self.syllables)
        print(f"Number of unique syllables: {self.vocab_size}")
        with open("/scratch2/bsow/Documents/ACSR/data/claire_dialogue/syllable_vocab.txt", "w", encoding="utf-8") as file:
            for syllable in self.syllables:
                file.write(syllable + "\n")
        self.data = self._create_sequences()

    def _get_syllables(self):
        all_syllables = [syllable for sentence in self.syllabized_ipa_sentences for syllable in sentence]
        syllable_counts = Counter(all_syllables)
        return sorted(syllable_counts.keys())

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
        input_indices = [self.syllable_to_idx[syllable] for syllable in input_seq]
        output_index = self.syllable_to_idx[output_seq]
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

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
            avg_val_loss = val_loss / len(val_loader)

        val_accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        sys.stdout.flush()
        wandb.log({"val_loss": avg_val_loss, "val_accuracy": val_accuracy, "epoch": epoch + 1})

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = os.path.join(save_dir, f"best_model_epoch_{epoch + 1}_accuracy_{val_accuracy:.4f}.pth")
            torch.save(model.state_dict(), model_save_path)
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "best_optimizer.pth"))
            print(f"Model saved to {model_save_path} with validation accuracy: {val_accuracy:.4f}")
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

    epochs, batch_size, hidden_dim, embedding_dim, num_layers, seq_length, learning_rate = 300, 16, 512, 256, 2, 5, 0.001
    dataset = SyllableDataset(syllabized_ipa_sentences, seq_length=6)
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