"""
Text processing utilities for the ACSR system.
This module contains functions for processing text, syllables, and phonemes.
"""

from ..variables import consonant_to_handshapes, vowel_to_position


def syllabify_liaphon(ipa_text):
    """
    Convert IPA text to syllables.
    
    Args:
        ipa_text (str): IPA text with space-separated phonemes.
        
    Returns:
        list: List of syllables.
    """
    consonants = {"b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "n~", "p", "r", "s", "s^", "t", "v", "w", "z", "z^", "ng", "gn"}
    vowels = {"a", "a~", "e", "e^", "e~", "i", "o", "o^", "o~", "u", "y", "x", "x^", "x~"}
    phonemes = ipa_text.split()
    syllables = []
    i = 0

    while i < len(phonemes):
        phone = phonemes[i]
        if phone in vowels:
            syllables.append(phone)
            i += 1
        elif phone in consonants:
            # Check if there is a next phone
            if i + 1 < len(phonemes):
                next_phone = phonemes[i + 1]
                if next_phone in vowels:
                    syllable = phone + next_phone
                    syllables.append(syllable)
                    i += 2 
                else:
                    if next_phone == "q":
                        syllables.append(phone)
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

import re

def convert_ipa_to_liaphon(ipa_text, ipa_to_target):
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

def syllables_to_gestures(syllable_sequence):
    """
    Convert a sequence of syllables into a sequence of gestures.
    
    Args:
        syllable_sequence (list): A list of syllables (strings).
        
    Returns:
        list: A list of gesture strings in the format "handshape-position".
    """
    gestures = []
    for syllable in syllable_sequence:
        if syllable == "<SOS>" or syllable == "<EOS>" or syllable == "<PAD>" or syllable == "<UNK>":
            gestures.append(syllable)
        # Check if the syllable starts with a multi-character consonant (e.g., "s^")
        elif len(syllable) >= 3 and syllable[:2] in consonant_to_handshapes:
            consonant = syllable[:2]
            vowel = syllable[2:]  # Remaining part is the vowel
            handshape = consonant_to_handshapes.get(consonant, 5)  # Default handshape is 5
            position = vowel_to_position.get(vowel, 1)  # Default position is 1
            gestures.append(f"{handshape}-{position}")
        # Check if the syllable ends with a multi-character vowel (e.g., "me^")
        elif len(syllable) >= 3 and syllable[-2:] in vowel_to_position:
            consonant = syllable[:-2]  # Remaining part is the consonant
            vowel = syllable[-2:]
            handshape = consonant_to_handshapes.get(consonant, 5)  # Default handshape is 5
            position = vowel_to_position.get(vowel, 1)  # Default position is 1
            gestures.append(f"{handshape}-{position}")
        # Handle normal CV syllables (e.g., "ma")
        elif len(syllable) == 2:
            if syllable in consonant_to_handshapes:  # length 2 consonant only syllable
                handshape = consonant_to_handshapes.get(syllable, 5)  # Default handshape is 5
                position = 1  # Default position is 1
                gestures.append(f"{handshape}-{position}")
            elif syllable in vowel_to_position:  # length 2 vowel only syllable
                handshape = 5  # Default handshape is 5
                position = vowel_to_position.get(syllable, 1)
                gestures.append(f"{handshape}-{position}")
            elif syllable[0] in consonant_to_handshapes:  # Consonant-Vowel pair
                consonant = syllable[0]
                vowel = syllable[1]
                handshape = consonant_to_handshapes.get(consonant, 5)  # Default handshape is 5
                position = vowel_to_position.get(vowel, 1)  # Default position is 1
                gestures.append(f"{handshape}-{position}")
            elif syllable[0] in vowel_to_position:  # Vowel-only syllable
                vowel = syllable
                position = vowel_to_position.get(vowel, 1)  # Default position is 1
                gestures.append(f"5-{position}")  # Default handshape is 5
        # Handle C-only syllables (e.g., "m")
        elif len(syllable) == 1 and syllable in consonant_to_handshapes:
            handshape = consonant_to_handshapes.get(syllable, 5)  # Default handshape is 5
            gestures.append(f"{handshape}-1")  # Default position is 1
        # Handle V-only syllables (e.g., "a")
        elif len(syllable) == 1 and syllable in vowel_to_position:
            position = vowel_to_position.get(syllable, 1)  # Default position is 1
            gestures.append(f"5-{position}")  # Default handshape is 5
        else:
            # Unknown syllable
            print(f"Unknown syllable: {syllable}")
    return gestures


def syllables_to_phonemes(syllable_sequence):
    """
    Convert a sequence of syllables into a sequence of phonemes.
    
    Args:
        syllable_sequence (list): A list of syllables (strings).
        
    Returns:
        list: A list of phonemes.
    """
    phonemes = []
    for syllable in syllable_sequence:
        if syllable == " ":
            phonemes.append(" ")
            continue
        if syllable == "<SOS>" or syllable == "<EOS>" or syllable == "<PAD>" or syllable == "<UNK>":
            phonemes.append(syllable)
            continue
        
        # Handle multi-character consonants (e.g., "s^")
        if len(syllable) >= 3 and syllable[:2] in consonant_to_handshapes:
            consonant = syllable[:2]
            vowel = syllable[2:]  # Remaining part is the vowel
            phonemes.append(consonant)
            phonemes.append(vowel)
        
        # Handle multi-character vowels (e.g., "me^")
        elif len(syllable) >= 3 and syllable[-2:] in vowel_to_position:
            consonant = syllable[:-2]  # Remaining part is the consonant
            vowel = syllable[-2:]
            phonemes.append(consonant)
            phonemes.append(vowel)
        
        # Handle normal CV syllables (e.g., "ma")
        elif len(syllable) == 2:
            consonant = syllable[0]
            vowel = syllable[1]
            phonemes.append(consonant)
            phonemes.append(vowel)
        
        # Handle C-only syllables (e.g., "m")
        elif len(syllable) == 1 and syllable in consonant_to_handshapes:
            phonemes.append(syllable)
        
        # Handle V-only syllables (e.g., "a")
        elif len(syllable) == 1 and syllable in vowel_to_position:
            phonemes.append(syllable)
        
        else:
            # Unknown syllable
            print(f"Unknown syllable: {syllable}")
    
    return phonemes 