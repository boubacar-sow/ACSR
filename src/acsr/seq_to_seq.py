import csv
import os
import sys
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pandas as pd
import numpy as np
import wandb
from decoding import syllables_to_gestures
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

# Configuration
CONFIG = {
    "data_path": "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/preprocessed_train.txt",
    "vocab_path": "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/vocab.txt",
    "seq_length": 50,
    "batch_size": 400,
    "embedding_dim": 128,
    "hidden_dim": 512,
    "num_layers": 3,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "epochs": 400,
    "teacher_forcing_initial": 0.8,   # initial teacher forcing ratio
    "teacher_forcing_min": 0.3,       # minimum teacher forcing ratio after annealing
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "special_tokens": ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
}

# Gesture mappings (provided by user)
consonant_to_handshapes = {
    "b": 4, "d": 1, "f": 5, "g": 7, "h": 4, "j": 8, 
    "k": 2, "l": 6, "m": 5, "n": 4, "p": 1, "r": 3, 
    "s": 3, "s^": 6, "t": 5, "v": 2, "w": 6, "z": 2, 
    "z^": 1, "ng": 6, "gn": 8
}

vowel_to_position = {
    "a": 1, "a~": 3, "e": 5, "e^": 4, "e~": 2,
    "i": 3, "o": 4, "o^": 1, "o~": 3, "u": 4, 
    "y": 5, "x": 1, "x^": 1, "x~": 5
}

# ---------------- Dataset & Collate ----------------
class Seq2SeqDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab
        self.max_token_id = len(vocab) - 1
        self.sos_token = vocab["<SOS>"]
        self.eos_token = vocab["<EOS>"]

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_tensor = self._sequence_to_tensor(src)
        tgt_tensor = self._sequence_to_tensor(tgt, is_target=True)
        return src_tensor, tgt_tensor, len(src_tensor), len(tgt_tensor)

    def _sequence_to_tensor(self, sequence, is_target=False):
        tokens = [self.vocab.get(token, self.max_token_id) for token in sequence]
        if is_target:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        return torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch, pad_token):
    src_batch, tgt_batch, src_lens, tgt_lens = zip(*batch)
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_token)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_token)
    
    # Create target mask for loss calculation (True for real tokens)
    tgt_mask = (tgt_padded != pad_token)
    
    return src_padded, tgt_padded, tgt_mask, torch.tensor(src_lens), torch.tensor(tgt_lens)

# ---------------- Attention Module ----------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch, hidden_dim]
        # encoder_outputs: [batch, src_len, hidden_dim]
        # mask: [batch, src_len] with 1 for real tokens, 0 for pad
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, src_len, hidden_dim]
        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        # Apply mask: set scores for padded tokens to a large negative value
        attention = attention.masked_fill(mask == 0, -1e10)
        return torch.softmax(attention, dim=1)  # [batch, src_len]

# ---------------- Encoder ----------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        
    def forward(self, src, src_mask):
        # src_mask: [batch, src_len] binary mask where 1 indicates a real token
        embedded = self.embedding(src)
        lengths = src_mask.sum(dim=1).cpu()  # compute actual lengths
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # Return outputs, hidden states, cell states, and lengths
        return outputs, hidden, cell, lengths

# ---------------- Decoder with Attention ----------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)
        self.attention = Attention(hidden_dim)
        # Combine LSTM output and context vector (hidden_dim*2)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x, hidden, cell, encoder_outputs, src_mask):
        # x: [batch, 1] current token input
        embedded = self.embedding(x)  # [batch, 1, embedding_dim]
        lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # [batch, 1, hidden_dim]
        
        # Use the last layer's hidden state for attention
        attn_weights = self.attention(hidden[-1], encoder_outputs, src_mask)  # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden_dim]
        
        combined = torch.cat((lstm_output, context), dim=2)  # [batch, 1, hidden_dim*2]
        prediction = self.fc(combined)  # [batch, 1, vocab_size]
        return prediction, hidden, cell, attn_weights

# ---------------- Seq2Seq Model ----------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        
    def forward(self, src, token_to_id, tgt=None, teacher_forcing_ratio=0.9, training=True):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1] if training else src.shape[1]+2
        tgt_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        # Create source mask from padded src: [batch, src_len]
        src_mask = (src != self.pad_idx)
        encoder_outputs, hidden, cell, lengths = self.encoder(src, src_mask)
        
        # Recompute mask based on actual encoder_outputs length
        max_len = encoder_outputs.size(1)
        # Build new mask: for each batch element, positions < length are True
        encoder_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.to(self.device).unsqueeze(1)
        
        sos_token = token_to_id["<SOS>"]
        decoder_input = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=self.device)
        # Track sequences that have reached EOS
        eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for t in range(1, tgt_len):
            # Decode one step
            output, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs, encoder_mask)
            outputs[:, t] = output.squeeze(1)
            if training:
                teacher_force = random.random() < teacher_forcing_ratio
                decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else output.argmax(2)
            else:
                decoder_input = output.argmax(2)
                new_eos_mask = (decoder_input.squeeze(1) == token_to_id["<EOS>"])
                eos_mask = eos_mask | new_eos_mask
                if eos_mask.all():
                    break

            decoder_input = torch.where(eos_mask.unsqueeze(1), token_to_id["<EOS>"], decoder_input)

        return outputs
# ---------------- Data Generation Functions ----------------
def generate_noisy_sequence(original_seq, vocab, vocab_list, max_changes=1):
    operations = ["modify", "drop", "add", "keep"]
    noisy_seq = original_seq.copy()
    
    num_changes = 1 if len(noisy_seq) > 40 else 2
        
    for _ in range(num_changes):
        op = random.choice(operations)
        
        if op == "modify" and len(noisy_seq) > 0:
            idx = random.randint(0, len(noisy_seq)-1)
            original_syllable = noisy_seq[idx]
            gesture = syllables_to_gestures([original_syllable])[0]
            
            if random.random() < 0.8 and gesture not in ["<UNK>", "<PAD>"]:
                candidates = [syl for syl in vocab_list 
                             if syllables_to_gestures([syl])[0] == gesture]
                if candidates:
                    noisy_seq[idx] = random.choice(candidates)
                    continue
            noisy_seq[idx] = random.choice(vocab_list)
            
        elif op == "drop" and len(noisy_seq) > 1:
            idx = random.randint(0, len(noisy_seq)-1)
            del noisy_seq[idx]
            
        elif op == "add":
            new_syllable = random.choice(vocab_list)
            idx = random.randint(0, len(noisy_seq))
            noisy_seq.insert(idx, new_syllable)
        elif op == "keep":
            pass

    if len(noisy_seq) == 0:
        noisy_seq = original_seq.copy()  # Fallback to original sequence
    return noisy_seq

def create_noisy_dataset(clean_sequences, vocab, vocab_list, num_samples_per_seq=1):
    pairs = []
    for seq in clean_sequences:
        if len(seq) > CONFIG["seq_length"] or len(seq) <= 1:  # Skip empty sequences
            continue
        for _ in range(num_samples_per_seq):
            noisy = generate_noisy_sequence(seq, vocab, vocab_list)
            if len(noisy) > 0:  # Ensure noisy sequence is not empty
                pairs.append((" ".join(noisy), " ".join(seq)))
    return pairs

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def load_vocab(vocab_path):
    with open(vocab_path) as f:
        vocab = [line.strip() for line in f]
    vocab = unique(vocab)
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    return token_to_id, vocab

def load_and_preprocess_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    clean_sequences = []
    for line in lines:
        line = line.strip().lower().replace("\n", "")
        if not line:
            continue
        syllables = line.split(' ')
        chunk = syllables
        if len(chunk) < 8:
            continue
        clean_sequences.append(chunk)

    print(f"Number of sequences: {len(clean_sequences)}")
    return clean_sequences


# ---------------- Training and Evaluation ----------------
def train_model(model, train_loader, val_loader, criterion, optimizer):
    best_loss = float('inf')
    initial_tf = CONFIG["teacher_forcing_initial"]
    tf_min = CONFIG["teacher_forcing_min"]
    total_epochs = CONFIG["epochs"]
    import time
    for epoch in range(CONFIG["epochs"]):
        start = time.time()
        # Anneal teacher forcing linearly over epochs
        teacher_forcing_ratio = max(tf_min, initial_tf - ((initial_tf - tf_min) * (epoch / total_epochs)))
        model.train()
        train_loss = 0
        for src, tgt, tgt_mask, _, _ in train_loader:
            src, tgt = src.to(CONFIG["device"]), tgt.to(CONFIG["device"])
            tgt_mask = tgt_mask.to(CONFIG["device"])
            
            optimizer.zero_grad()
            output = model(src, tgt, teacher_forcing_ratio)
            
            # Masked loss calculation
            output_dim = output.shape[-1]
            loss = criterion(output.view(-1, output_dim), tgt.view(-1))
            loss = (loss.view(tgt.shape) * tgt_mask).sum() / tgt_mask.sum()
            
            loss.backward()
            # Optional: gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        val_loss = evaluate_model(model, val_loader, criterion)
        end = time.time()
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Teacher Forcing: {teacher_forcing_ratio:.3f} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Time: {round(end - start, 2)} seq")
        sys.stdout.flush()
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "/pasteur/appa/homes/bsow/ACSR/src/acsr/saved_models/best_seq2seq_model_last.pth")

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt, tgt_mask, _, _ in loader:
            src, tgt = src.to(CONFIG["device"]), tgt.to(CONFIG["device"])
            tgt_mask = tgt_mask.to(CONFIG["device"])
            
            output = model(src, None, teacher_forcing_ratio=0, training=False)  # no teacher forcing in eval
            output_dim = output.shape[-1]
            loss = criterion(output.view(-1, output_dim), tgt.view(-1))
            loss = (loss.view(tgt.shape) * tgt_mask).sum() / tgt_mask.sum()
            total_loss += loss.item()
    return total_loss / len(loader)

def decode_sequence(tensor, idx_to_token):
    """Convert tensor of indices to syllable string"""
    tokens = []
    for idx in tensor:
        token = idx_to_token[idx.item()]
        if token == "<EOS>":
            break
        if token not in ["<PAD>", "<SOS>"]:
            tokens.append(token)
    return " ".join(tokens)

def test_examples(model, dataset, idx_to_token, num_examples=10, device="cpu"):
    model.eval()
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    for idx in indices:
        src, tgt, _, _ = dataset[idx]
        src_tensor = src.unsqueeze(0).to(device)
        
        # Generate prediction
        with torch.no_grad():
            output = model(src_tensor, tgt.unsqueeze(0).to(device), teacher_forcing_ratio=0)
            pred = output.argmax(-1).squeeze()
        
        # Convert to strings
        src_str = decode_sequence(src, idx_to_token)
        tgt_str = decode_sequence(tgt, idx_to_token)
        pred_str = decode_sequence(pred, idx_to_token)
        
        print(f"\nExample {idx+1}:")
        print(f"Source:  {src_str}")
        print(f"Target:  {tgt_str}")
        print(f"Predict: {pred_str}")
        print("="*80)
        sys.stdout.flush()

# ---------------- Main ----------------
if __name__ == "__main__":
    # Load vocabulary
    token_to_id, vocab_list = load_vocab(CONFIG["vocab_path"])
    
    # Load and preprocess data
    clean_sequences = load_and_preprocess_data(CONFIG["data_path"])
    print(f"Number of clean sequences: {len(clean_sequences)}")
    sys.stdout.flush()

    noisy_pairs_path = "/pasteur/appa/homes/bsow/ACSR/data/french_dataset/noisy_pairs.csv"
    if os.path.exists(noisy_pairs_path):
        print("Loading noisy dataset from file...")
        sys.stdout.flush()
        pairs = []
        with open(noisy_pairs_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # Skip header
            for row in reader:
                noisy, clean = row
                if len(noisy) > 0 and len(clean) > 0:
                    pairs.append((noisy.split(' '), clean.split(' ')))
    else:
        print("Generating noisy dataset...")
        sys.stdout.flush()
        pairs = create_noisy_dataset(clean_sequences, token_to_id, vocab_list)
        with open(noisy_pairs_path, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["noisy", "clean"])
            writer.writerows([(noisy, clean) for noisy, clean in pairs])

    print("Pairs created. Number of pairs:", len(pairs))
    #train_pairs, val_pairs = train_test_split(pairs, test_size=0.1)

    indices_file = '/pasteur/appa/homes/bsow/ACSR/src/acsr/indices.pkl'
    if os.path.exists(indices_file):
        with open(indices_file, 'rb') as f:
            indices = pickle.load(f)
    else:
        indices = list(range(len(pairs)))
        random.shuffle(indices)
        
        with open(indices_file, 'wb') as f:
            pickle.dump(indices, f)

    # Split the pairs into training and validation sets using the indices
    train_pairs = [pairs[i] for i in indices[:int(len(indices) * 0.9)] if len(pairs[i]) > 0]
    val_pairs = [pairs[i] for i in indices[int(len(indices) * 0.9):] if len(pairs[i]) > 0]

    print("Number of training pairs:", len(train_pairs))
    print("Number of validation pairs:", len(val_pairs))
    sys.stdout.flush()
    train_dataset = Seq2SeqDataset(train_pairs, token_to_id)
    val_dataset = Seq2SeqDataset(val_pairs, token_to_id)
    
    pad_token = token_to_id["<PAD>"]
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
        collate_fn=lambda batch: collate_fn(batch, pad_token), pin_memory=True, num_workers=6
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
        collate_fn=lambda batch: collate_fn(batch, pad_token), pin_memory=True, num_workers=6
    )
    
    encoder = Encoder(len(token_to_id), CONFIG["embedding_dim"], 
                      CONFIG["hidden_dim"], CONFIG["num_layers"], CONFIG["dropout"])
    decoder = Decoder(len(token_to_id), CONFIG["embedding_dim"],
                      CONFIG["hidden_dim"], CONFIG["num_layers"], CONFIG["dropout"])
    model = Seq2Seq(encoder, decoder, CONFIG["device"], pad_token).to(CONFIG["device"])
    #model.load_state_dict(torch.load("best_seq2seq_model_long.pth"))
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_id["<PAD>"], reduction='none')
    
    train_model(model, train_loader, val_loader, criterion, optimizer)

    idx_to_token = {v: k for k, v in token_to_id.items()}

    print("\nTesting on Validation Examples:")
    test_examples(model, val_dataset, idx_to_token, num_examples=10, device=CONFIG["device"])

    print("\nTesting on Training Examples:")
    test_examples(model, train_dataset, idx_to_token, num_examples=10, device=CONFIG["device"])
