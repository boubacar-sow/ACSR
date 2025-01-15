import os
import pandas as pd
import subprocess
import re  # For filtering IPA symbols

# Paths
common_voice_dir = "/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/fr"
ipa_dir = "/scratch2/bsow/Documents/ACSR/data/cv-corpus-20.0-delta-2024-12-06/ipa"
clips_dir = os.path.join(common_voice_dir, "clips")
validated_tsv = os.path.join(common_voice_dir, "validated.tsv")

# Create IPA directory if it doesn't exist
os.makedirs(ipa_dir, exist_ok=True)

# Define a regex pattern for valid IPA symbols
# This pattern can be adjusted based on the IPA symbols you expect
ipa_pattern = re.compile(r"[a-zA-Zæɑɒɔəɛɜɪɲʃʒɛ̃ʁɔ̃ɑ̃eɡ]+")

# Function to filter out non-IPA symbols
def filter_ipa_symbols(text):
    """
    Filter out non-IPA symbols from the text.
    Args:
        text (str): The input text.
    Returns:
        str: Text containing only IPA symbols.
    """
    # Find all matches of IPA symbols
    ipa_symbols = ipa_pattern.findall(text)
    # Join the matches into a single string
    return " ".join(ipa_symbols)

# Function to convert text to IPA phones using espeak-ng
def phonemize_text_with_espeak(text):
    """
    Convert text to IPA phones using espeak-ng and split into individual phones.
    Args:
        text (str): The input text.
    Returns:
        str: Space-separated IPA phones without stress markers or hyphens.
    """
    # Use espeak-ng to convert text to IPA
    command = ["espeak-ng", "-v", "fr", "-q", "--ipa"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=text.encode())
    
    if process.returncode != 0:
        raise RuntimeError(f"espeak-ng failed: {stderr.decode()}")
    
    # Decode and clean the IPA output
    ipa_output = stdout.decode().strip()
    
    # Remove stress markers (ˈ and ˌ), hyphens (-), and newlines
    ipa_output = ipa_output.replace("ˈ", "").replace("ˌ", "").replace("-", "").replace("\n", " ")
    
    # Filter out non-IPA symbols
    ipa_output = filter_ipa_symbols(ipa_output)
    
    # Split into individual phones and join with spaces
    ipa_phones = " ".join(list(ipa_output))
    return ipa_phones

# Function to process a single row
def process_row(row):
    clip_id = row["path"].replace(".mp3", "")
    sentence = row["sentence"]
    ipa_output_path = os.path.join(ipa_dir, f"{clip_id}.txt")

    try:
        # Step 1: Convert sentence to IPA phones using espeak-ng
        ipa_phones = phonemize_text_with_espeak(sentence)

        # Step 2: Save the IPA phones to a file (ensure single line)
        with open(ipa_output_path, "w") as f:
            f.write(ipa_phones.strip())  # Ensure no extra newlines
        print(f"IPA phones saved to: {ipa_output_path}")
    except Exception as e:
        print(f"Error processing {clip_id}: {e}")

# Main function
def main():
    # Load the validated.tsv file
    validated_df = pd.read_csv(validated_tsv, sep='\t')

    # Process all rows in the validated.tsv file
    for _, row in validated_df.iterrows():
        process_row(row)

if __name__ == "__main__":
    main()