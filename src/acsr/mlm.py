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
import wandb
import multiprocessing
import sys
from functools import partial
from decoder_lm import clean_text, text_to_ipa, syllabify_ipa, convert_ipa_to_syllables

# ------------------------------
# Configuration
# ------------------------------
CONFIG = { 
    "preprocessed_train_path": "/scratch2/bsow/Documents/ACSR/data/french_dataset/preprocessed_train.txt",
    "preprocessed_val_path": "/scratch2/bsow/Documents/ACSR/data/french_dataset/preprocessed_val.txt",
    "data_train_path": "/scratch2/bsow/Documents/ACSR/data/french_dataset/concat_train.csv",
    "data_val_path": "/scratch2/bsow/Documents/ACSR/data/french_dataset/concat_val.csv",
    "preprocessed_path": "/scratch2/bsow/Documents/ACSR/data/news_dataset/preprocessed_syllables.txt",
    "batch_size": 1024,
    "embedding_dim": 200,
    "hidden_dim": 300,
    "num_layers": 4,    # Number of transformer encoder layers
    "dropout": 0.2,
    "learning_rate": 0.001,
    "epochs": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vocab_size": 270,  # This will be updated after building the vocab
    "min_count": 1,
    # Added <MASK> token for MLM
    "special_tokens": ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<MASK>"],
    "mask_prob": 0.15,  # Probability of masking a token
    "max_seq_length": 128,  # Maximum sequence length for the transformer
    "nhead": 4  # Number of attention heads in the transformer
}

# ------------------------------
# Phoneme mapping 
# ------------------------------
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
# Preprocessing Helpers
# ------------------------------
def save_preprocessed(data, path):
    """Save preprocessed data with space-separated syllables"""
    with open(path, 'w', encoding='utf-8') as f:
        for syllables in data:
            f.write(' '.join(syllables) + '\n')

def load_preprocessed(path):
    """Load preprocessed data and split into syllable lists"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().split(' ') for line in f]

def load_and_clean_data(file_path):
    """Load and clean data from a CSV file (assumes a 'text' column)"""
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

def preprocess_data():
    """Load and preprocess both training and validation data, with caching"""
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
    """Build vocabulary from training data (plus special tokens)"""
    counter = Counter()
    for syllables in train_syllables:
        counter.update(syllables)
    # Start with special tokens
    vocab = CONFIG['special_tokens'].copy()
    # Optionally, load additional syllables from an external file
    with open("/scratch2/bsow/Documents/ACSR/data/news_dataset/vocab.txt", "r") as f:
        syllables = f.readlines()
        syllables = [syl.replace(" ", "").replace("\n", "") for syl in syllables]
    vocab += syllables
    vocab = unique(vocab)
    return {syl: idx for idx, syl in enumerate(vocab)}, vocab

# ------------------------------
# Masked Language Modeling Dataset
# ------------------------------
class MaskedSyllableDataset(Dataset):
    def __init__(self, sequences, syllable_to_idx, mask_prob=0.15):
        """
        sequences: list of token lists (each a full sentence)
        syllable_to_idx: vocabulary mapping
        mask_prob: probability to mask a token (except special tokens)
        """
        self.sequences = sequences
        self.syllable_to_idx = syllable_to_idx
        self.mask_prob = mask_prob
        self.mask_token = "<MASK>"
        self.mask_token_id = self.syllable_to_idx[self.mask_token]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Get the token list for the sentence and add <SOS> and <EOS>
        tokens = ['<SOS>'] + self.sequences[idx] + ['<EOS>']
        input_tokens = tokens.copy()
        labels = [-100] * len(tokens)  # -100 will be ignored in the loss
        
        # For each token (except special tokens), possibly mask it
        for i in range(len(tokens)):
            if tokens[i] in ["<SOS>", "<EOS>", "<PAD>", "<MASK>"]:
                continue
            if torch.rand(1).item() < self.mask_prob:
                # Set the label for this position to the original token id
                labels[i] = self.syllable_to_idx.get(tokens[i], self.syllable_to_idx["<UNK>"])
                prob = torch.rand(1).item()
                if prob < 0.8:
                    input_tokens[i] = "<MASK>"
                elif prob < 0.9:
                    # Replace with a random token from the vocabulary
                    token_list = list(self.syllable_to_idx.keys())
                    input_tokens[i] = token_list[torch.randint(len(token_list), (1,)).item()]
                else:
                    # Leave the token unchanged (but still predict it)
                    pass
        # Convert tokens to ids
        input_ids = [self.syllable_to_idx.get(token, self.syllable_to_idx["<UNK>"]) for token in input_tokens]
        return torch.tensor(input_ids), torch.tensor(labels)

def create_mlm_datasets(train_syllables, val_syllables, syllable_to_idx, mask_prob=0.15):
    """
    Create MLM datasets by simply using the full token sequences.
    """
    # Here, each element in train_syllables/val_syllables is a list of syllables.
    # The dataset itself will add <SOS> and <EOS> and perform masking.
    return (
        MaskedSyllableDataset(train_syllables, syllable_to_idx, mask_prob),
        MaskedSyllableDataset(val_syllables, syllable_to_idx, mask_prob)
    )

# ------------------------------
# Collate Function for Padding
# ------------------------------
def get_mlm_collate_fn(pad_token_id):
    def collate_fn(batch):
        # batch is a list of (input_ids, labels), each of variable length
        input_ids, labels = zip(*batch)
        # Determine the maximum length in the batch, but cap it at max_seq_length.
        max_len = min(max(x.size(0) for x in input_ids), CONFIG["max_seq_length"])
        padded_inputs = []
        padded_labels = []
        for inp, lab in zip(input_ids, labels):
            # Truncate if necessary
            inp = inp[:max_len]
            lab = lab[:max_len]
            pad_length = max_len - inp.size(0)
            padded_inputs.append(torch.cat([inp, torch.full((pad_length,), pad_token_id, dtype=torch.long)]))
            padded_labels.append(torch.cat([lab, torch.full((pad_length,), -100, dtype=torch.long)]))
        return torch.stack(padded_inputs), torch.stack(padded_labels)
    return collate_fn


# ------------------------------
# Transformer-based Masked Language Model
# ------------------------------
class MaskedLMTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, dropout, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.size()
        # Create position ids (0-indexed)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        token_embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        pos_embeddings = self.pos_embedding(position_ids)  # [batch_size, seq_len, embedding_dim]
        embeddings = token_embeddings + pos_embeddings
        embeddings = self.dropout(embeddings)
        # Transformer expects shape: [seq_len, batch_size, embedding_dim]
        embeddings = embeddings.transpose(0, 1)
        transformer_output = self.transformer_encoder(embeddings)  # [seq_len, batch_size, embedding_dim]
        transformer_output = transformer_output.transpose(0, 1)  # [batch_size, seq_len, embedding_dim]
        logits = self.fc(transformer_output)  # [batch_size, seq_len, vocab_size]
        return logits

# ------------------------------
# Training Pipeline
# ------------------------------
def train_model(train_dataset, val_dataset, model, pad_token_id):
    wandb.init(project="ipa-syllable-mlm", config=CONFIG)
    
    collate_fn = get_mlm_collate_fn(pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        start = time.time()
        model.train()
        train_loss = 0.0
        total_tokens = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            optimizer.zero_grad()
            outputs = model(inputs)  # [batch_size, seq_len, vocab_size]
            # Reshape for loss computation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Count tokens where target != -100
            num_tokens = (targets != -100).sum().item()
            train_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_train_loss = train_loss / total_tokens
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        total_val_tokens = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
                num_tokens = (targets != -100).sum().item()
                val_loss += loss.item() * num_tokens
                total_val_tokens += num_tokens
        avg_val_loss = val_loss / total_val_tokens
        
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        end = time.time()
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Perplexity: {perplexity:.2f} | "
              f"Time: {end-start:.2f}s")
        sys.stdout.flush()
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "perplexity": perplexity
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_syllable_mlm_model.pth")
            wandb.save("best_syllable_mlm_model.pth")

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
    
    # Create MLM datasets
    train_dataset, val_dataset = create_mlm_datasets(train_syllables, val_syllables, syllable_to_idx, CONFIG["mask_prob"])
    # Prepare the collate function using the <PAD> token id
    pad_token_id = syllable_to_idx["<PAD>"]
    
    # Initialize the transformer-based MLM model
    model = MaskedLMTransformer(
        vocab_size=len(vocab),
        embedding_dim=CONFIG['embedding_dim'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        max_seq_length=CONFIG['max_seq_length']
    ).to(CONFIG['device'])
    
    # Login to wandb (ensure your API key is set up securely)
    wandb.login(key="580ab03d7111ed25410f9831b06b544b5f5178a2")
    train_model(train_dataset, val_dataset, model, pad_token_id)
    
    # Example of model predictions (inference)
    model.eval()
    import numpy as np
    np.random.seed(42)
    # Pick a random sample from the validation set
    sample_input, sample_labels = val_dataset[np.random.randint(len(val_dataset))]
    sample_input = sample_input.unsqueeze(0).to(CONFIG['device'])
    with torch.no_grad():
        logits = model(sample_input)
        probabilities = torch.softmax(logits, dim=-1)
    # For demonstration, print the input tokens, the masked positions (where label != -100),
    # and the top prediction for each masked token.
    input_tokens = [idx_to_syllable[idx.item()] for idx in sample_input[0]]
    label_tokens = [idx_to_syllable[label.item()] if label.item() != -100 else "_" for label in sample_labels]
    top_preds = torch.topk(probabilities, k=1, dim=-1).indices[0].squeeze(-1)
    pred_tokens = [idx_to_syllable[idx.item()] for idx in top_preds]
    
    print("\nSample Prediction:")
    print("Input:  ", input_tokens)
    print("Target: ", label_tokens)
    print("Predicted: ", pred_tokens)
