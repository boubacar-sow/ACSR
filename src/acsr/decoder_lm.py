import re
import subprocess


# Path to the dataset
file_path = "/scratch2/bsow/Documents/ACSR/data/claire_dialogue/train.txt"

# Function to clean the text
def clean_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    text = re.sub(r"\[.*?\]\s*", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    cleaned_text = []
    current_sentence = ""

    for line in lines:
        line = line.lower()
        cleaned_text.append(line)

    if current_sentence:
        cleaned_text.append(current_sentence.strip())

    return cleaned_text

# Function to remove punctuation from a list of sentences
def remove_punctuation(sentences):
    punctuation_pattern = re.compile(r"[^\w\s'-]")  
    cleaned_sentences = []

    for sentence in sentences:
        # Remove punctuation using the regex pattern
        cleaned_sentence = re.sub(punctuation_pattern, "", sentence)
        cleaned_sentences.append(cleaned_sentence.strip())

    return cleaned_sentences

# Function to convert text to IPA using espeak-ng
def text_to_ipa(text, language="fr"):
    """
    Convert text to IPA using espeak-ng.
    """
    # Remove special characters
    #text = text.replace("?", "").replace("!", "").replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("'", "").replace("-", " ")

    command = ["espeak-ng", "-v", language, "-q", "--ipa"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=text.encode())
    ipa_output = stdout.decode().strip()
    ipa_output = ipa_output.replace("ˈ", "").replace("ˌ", "").replace("-", "").replace("\n", " ")

    return ipa_output

# Function to syllabify IPA text
def syllabify_ipa(ipa_text):
    consonants = "ptkbdgmnlrsfvzʃʒɡʁjwŋtrɥgʀycɲ"
    vowels = "aeɛioɔuøœəɑ̃ɛ̃ɔ̃œ̃ɑ̃ɔ̃ɑ̃ɔ̃"
    phonemes = list(ipa_text.replace(" ", ""))
    syllables = []
    i = 0

    while i < len(phonemes):
        phone = phonemes[i]
        if phone in vowels:
            # Check if the next character is a combining diacritic
            if i + 1 < len(phonemes) and phonemes[i + 1] == "̃":  # Corrected: No space after tilde
                syllable = phone + phonemes[i + 1]  # Combine base character with diacritic
                syllables.append(syllable)
                i += 2  # Skip the diacritic in the next iteration
            else:
                syllables.append(phone)
                i += 1
        elif phone in consonants:
            # Check if there is a next phone
            if i + 1 < len(phonemes):
                next_phone = phonemes[i + 1]
                if next_phone in vowels:
                    # Check if the vowel has a combining diacritic
                    if i + 2 < len(phonemes) and phonemes[i + 2] == "̃":  # Corrected: No space after tilde
                        syllable = phone + next_phone + phonemes[i + 2]  # Combine consonant, vowel, and diacritic
                        syllables.append(syllable)
                        i += 3  # Skip the diacritic in the next iteration
                    else:
                        syllable = phone + next_phone
                        syllables.append(syllable)
                        i += 2
                else:
                    syllables.append(phone)
                    i += 1
            else:
                syllables.append(phone)
                i += 1
        else:
            i += 1

    return syllables

def text_to_ipa(text, language="fr"):
    """
    Convert text to IPA using espeak-ng.
    """
    # Remove special characters
    #text = text.replace("?", "").replace("!", "").replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("'", "").replace("-", " ")

    command = ["espeak-ng", "-v", language, "-q", "--ipa"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=text.encode())
    ipa_output = stdout.decode().strip()
    ipa_output = ipa_output.replace("ˈ", "").replace("ˌ", "").replace("-", "").replace("\n", " ").replace("(en)", "").replace("(fr)", "")

    return ipa_output



ipa_to_target = {
    # Vowels
    "a": "a", "ɑ": "a", "ə": "x", "ɛ": "e^", "ø": "x", "œ": "x^", "i": "i", "y": "y", "e": "e",
    "u": "u", "ɔ": "o", "o": "o^", "ɑ̃": "a~", "ɛ̃": "e~", "ɔ̃": "o~", "œ̃": "x~",
    " ": " ",  # Space

    # Consonants
    "b": "b", "c": "k", "d": "d", "f": "f", "ɡ": "g", "j": "j", "k": "k", "l": "l", 
    "m": "m", "n": "n", "p": "p", "s": "s", "t": "t", "v": "v", "w": "w", "z": "z", 
    "ɥ": "h", "ʁ": "r", "ʃ": "s^", "ʒ": "z^", "ɲ": "gn", "ŋ": "ng"
}

import re

def convert_ipa_to_syllables(ipa_text, ipa_to_target):
    # Step 1: Convert IPA phonemes to target phonemes
    converted_text = []
    i = 0
    while i < len(ipa_text):
        char = ipa_text[i]
        # Check if the next character is a combining diacritic
        if i + 1 < len(ipa_text) and ipa_text[i + 1] == "̃":
            # Combine the base character with the diacritic
            combined_char = char + ipa_text[i + 1]
            # Map the combined character if it exists in the dictionary
            mapped_char = ipa_to_target.get(combined_char, combined_char)
            converted_text.append(mapped_char)
            i += 2  # Skip the diacritic in the next iteration
        else:
            # Map the single character
            mapped_char = ipa_to_target.get(char, "<UNK>")
            converted_text.append(mapped_char)
            i += 1
    return "".join(converted_text)


import multiprocessing

# Function to convert text to IPA using espeak-ng (modified for multiprocessing)
def text_to_ipa_worker(sentence):
    """
    Worker function for multiprocessing to convert text to IPA.
    """
    return text_to_ipa(sentence)

# Function to syllabify IPA text (modified for multiprocessing)
def syllabify_ipa_worker(ipa_sentence):
    """
    Worker function for multiprocessing to syllabify IPA text.
    """
    syllables = syllabify_ipa(ipa_sentence)
    new_syllables = convert_ipa_to_syllables(" ".join(syllables), ipa_to_target)
    if "<UNK>" in new_syllables:
        return []
    return new_syllables

if __name__ == "__main__":
    # load ipa sentences
    ipa_path = "/scratch2/bsow/Documents/ACSR/data/claire_dialogue/ipa_train.txt"
    ipa_sentences = []
    with open(ipa_path, "r", encoding="utf-8") as file:
        for line in file:
            ipa_sentences.append(line.strip())

    # Use multiprocessing to syllabify IPA sentences
    with multiprocessing.Pool(15) as pool:
        syllabized_ipa_sentences = pool.map(syllabify_ipa_worker, ipa_sentences)

    # Save the syllabized IPA sentences to a file
    syllabized_ipa_output_path = "/scratch2/bsow/Documents/ACSR/data/claire_dialogue/syllabized_ipa_train.txt"
    with open(syllabized_ipa_output_path, "w", encoding="utf-8") as file:
        for syllables in syllabized_ipa_sentences:
            if len(syllables) > 0:
                file.write("".join(syllables) + "\n")

    print(f"Syllabized IPA sentences saved to {syllabized_ipa_output_path}")