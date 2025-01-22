import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


def text_to_ipa(text, language="fr"):
    """
    Convert text to IPA using espeak-ng.
    """
    # remove special characters
    text = text.replace("?", "").replace("!", "").replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("'", "").replace("-", " ")

    command = ["espeak-ng", "-v", "fr", "-q", "--ipa"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=text.encode())
    ipa_output = stdout.decode().strip()
    ipa_output = ipa_output.replace("ˈ", "").replace("ˌ", "").replace("-", "").replace("\n", " ")
    
    return ipa_output


def split_ipa(ipa_text):
    """
    Split IPA text into individual phonemes, removing stress markers.
    """
    # Remove the apostrophe (stress marker)
    ipa_text = ipa_text.replace("ˈ", "").replace("ˌ", "")  # Remove primary and secondary stress markers
    # Split into individual phonemes
    phonemes = list(ipa_text)
    return phonemes


def syllabify_ipa(ipa_text):
    """
    Construct syllables from IPA text using a manual approach.
    Args:
        ipa_text (str): IPA text (e.g., "bɔ̃ʒuʁ").
    Returns:
        list: A list of syllables.
    """
    # Define Cued Speech consonants and vowels
    consonants = "ptkbdgmnlrsfvzʃʒɡʁjwŋtrɥgʀycɲ"
    vowels = "aeɛioɔuøœəɑ̃ɛ̃ɔ̃œ̃ɑ̃ɔ̃ɑ̃ɔ̃"

    # Remove spaces and split into individual phonemes
    phonemes = list(ipa_text.replace(" ", ""))

    # Construct syllables
    syllables = []
    i = 0

    while i < len(phonemes):
        phone = phonemes[i]

        # If the current phone is a vowel, treat it as a syllable
        if phone in vowels:
            syllables.append(phone)
            i += 1

        # If the current phone is a consonant, check the next phone
        elif phone in consonants:
            # Check if there is a next phone
            if i + 1 < len(phonemes):
                next_phone = phonemes[i + 1]

                # If the next phone is a vowel, combine into a CV syllable
                if next_phone in vowels:
                    syllable = phone + next_phone
                    syllables.append(syllable)
                    i += 2  # Skip the next phone since it's part of the syllable

                # If the next phone is not a vowel, treat the consonant as a standalone syllable
                else:
                    syllables.append(phone)
                    i += 1

            # If there is no next phone, treat the consonant as a standalone syllable
            else:
                syllables.append(phone)
                i += 1

        # If the phone is neither a consonant nor a vowel, skip it
        else:
            #print("Skipping phone:", phone)
            i += 1

    return syllables


def prepare_dataset(texts):
    """
    Prepare a dataset of phoneme-syllable pairs.
    """
    dataset = []
    for text in texts:
        ipa_text = text_to_ipa(text)
        syllables = syllabify_ipa(ipa_text)
        phonemes = split_ipa(ipa_text)  # Split IPA text into phonemes
        dataset.append((phonemes, syllables))
    return dataset


class PhonemeToSyllableDataset(Dataset):
    def __init__(self, dataset, phoneme_to_idx, syllable_to_idx):
        self.dataset = dataset
        self.phoneme_to_idx = phoneme_to_idx
        self.syllable_to_idx = syllable_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        phonemes, syllables = self.dataset[idx]
        phoneme_indices = [self.phoneme_to_idx[p] for p in phonemes]
        syllable_indices = [self.syllable_to_idx[s] for s in syllables]
        return torch.tensor(phoneme_indices, dtype=torch.long), torch.tensor(syllable_indices, dtype=torch.long)


def custom_collate_fn(batch):
    """
    Collate function to pad sequences to the same length within a batch.
    Args:
        batch: A list of tuples (phoneme_indices, syllable_indices).
    Returns:
        Padded phoneme and syllable sequences, along with their lengths.
    """
    # Separate phoneme and syllable sequences
    phoneme_sequences = [item[0] for item in batch]
    syllable_sequences = [item[1] for item in batch]

    # Pad sequences to the same length
    phoneme_padded = pad_sequence(phoneme_sequences, batch_first=True, padding_value=0)
    syllable_padded = pad_sequence(syllable_sequences, batch_first=True, padding_value=0)

    # Get lengths of sequences (before padding)
    phoneme_lengths = torch.tensor([len(seq) for seq in phoneme_sequences], dtype=torch.long)
    syllable_lengths = torch.tensor([len(seq) for seq in syllable_sequences], dtype=torch.long)

    return phoneme_padded, syllable_padded, phoneme_lengths, syllable_lengths


# Example long text
texts = [
    "Bonjour, comment ça va aujourd'hui? J'espère que tout va bien pour toi.",
    "La météo est magnifique aujourd'hui, le soleil brille et il fait chaud."
]

dataset = prepare_dataset(texts)
# Example tokenization
phoneme_to_idx = {p: i for i, p in enumerate(set(p for phonemes, _ in dataset for p in phonemes))}
syllable_to_idx = {s: i for i, s in enumerate(set(s for _, syllables in dataset for s in syllables))}

# Create index_to_syllable mapping
index_to_syllable = {idx: syllable for syllable, idx in syllable_to_idx.items()}

# Create dataset and dataloader
train_dataset = PhonemeToSyllableDataset(dataset, phoneme_to_idx, syllable_to_idx)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)


class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim=64):
        super(Seq2SeqModel, self).__init__()
        # Embedding layers
        self.phoneme_embedding = nn.Embedding(input_dim, embedding_dim)
        self.syllable_embedding = nn.Embedding(output_dim, embedding_dim)

        # Encoder and Decoder
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, src_lengths, tgt, tgt_lengths):
        # Embed the input sequences
        src_embedded = self.phoneme_embedding(src)  # (batch_size, seq_len, embedding_dim)
        tgt_embedded = self.syllable_embedding(tgt)  # (batch_size, seq_len, embedding_dim)

        # Pack padded sequences
        packed_src = nn.utils.rnn.pack_padded_sequence(src_embedded, src_lengths, batch_first=True, enforce_sorted=False)
        encoder_outputs, (hidden, cell) = self.encoder(packed_src)

        # Unpack packed sequences
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)

        # Pack padded target sequences
        packed_tgt = nn.utils.rnn.pack_padded_sequence(tgt_embedded, tgt_lengths, batch_first=True, enforce_sorted=False)
        decoder_outputs, _ = self.decoder(packed_tgt, (hidden, cell))

        # Unpack packed sequences
        decoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(decoder_outputs, batch_first=True)

        # Output layer
        output = self.fc(decoder_outputs)  # (batch_size, seq_len, output_dim)
        return output


# Training loop
model = Seq2SeqModel(input_dim=len(phoneme_to_idx), output_dim=len(syllable_to_idx), hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(500):
    for src, tgt, src_lengths, tgt_lengths in train_loader:
        optimizer.zero_grad()
        output = model(src, src_lengths, tgt, tgt_lengths)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Testing the model
# Set the model to evaluation mode
model.eval()

# Example text
text = "Bonjour"

# Convert text to IPA
ipa_text = text_to_ipa(text)

# Split IPA text into phonemes
phonemes = split_ipa(ipa_text)

# Convert phonemes to indices
phoneme_indices = torch.tensor([phoneme_to_idx[p] for p in phonemes], dtype=torch.long).unsqueeze(0)  # Add batch dimension

# Get length of the sequence
phoneme_lengths = torch.tensor([len(phonemes)], dtype=torch.long)

# Pass the phonemes through the model
with torch.no_grad():
    # Use dummy target input (not used during inference)
    dummy_tgt = torch.zeros_like(phoneme_indices)
    dummy_tgt_lengths = phoneme_lengths

    # Get model predictions
    output = model(phoneme_indices, phoneme_lengths, dummy_tgt, dummy_tgt_lengths)

    # Convert output to predicted syllable indices
    predicted_syllable_indices = torch.argmax(output, dim=-1)  # (batch_size, seq_len)

# Convert predicted indices to syllables
predicted_syllables = []
for i in range(predicted_syllable_indices.size(0)):
    syllable_indices = predicted_syllable_indices[i].tolist()
    syllables = [index_to_syllable[idx] for idx in syllable_indices if idx != 0]  # Skip padding
    predicted_syllables.append(syllables)

# Print results
print(f"Input Text: {text}")
print(f"IPA Transcription: {ipa_text}")
print(f"Input Phonemes: {phonemes}")
print(f"Predicted Syllables: {predicted_syllables}")