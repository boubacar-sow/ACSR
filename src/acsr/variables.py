"""
Variables module for the ACSR system.
This module contains constants and mappings used across the system.
"""

# Define the mapping of consonants to hand shapes
consonant_to_handshape = {
    "p": 1, "t": 5, "k": 2, "b": 4, "d": 1, "g": 7, "m": 5, "n": 4,
    "l": 6, "r": 3, "s": 3, "f": 5, "v": 2, "z": 2, "ʃ": 6, "ʒ": 1,
    "ɡ": 7, "ʁ": 3, "j": 8, "w": 6, "ŋ": 8, "ɥ": 4, "ʀ": 3, "y": 8, "c": 2
}

# Define vowel positions relative to the nose (right side of the face/body)
vowel_positions = {
    # Position -1: /a/, /o/, /œ/, /ə/
    "a": 1,
    "o": 1,
    "œ": 1,
    "ə": 1,

    # Position 50: /ɛ̃/, /ø/
    "ɛ̃": 2,
    "ø": 2,

    # Position 57: /i/, /ɔ̃/, /ɑ̃/
    "i": 3,
    "ɔ̃": 3,
    "ɑ̃": 3,

    # Position 175: /u/, /ɛ/, /ɔ/
    "u": 4,
    "ɛ": 4,
    "ɔ": 4,

    # Position -2: /œ̃/, /y/, /e/
    "œ̃": 5,
    "y": 5,
    "e": 5,
}

# Dictionary mapping consonants to handshape numbers, LIAPHON notation
consonant_to_handshapes = {
    "b": 4, 
    "d": 1, 
    "f": 5, 
    "g": 7, 
    "h": 4, 
    "j": 8, 
    "k": 2, 
    "l": 6, 
    "m": 5, 
    "n": 4, 
    "p": 1, 
    "r": 3, 
    "s": 3, 
    "s^": 6,
    "t": 5, 
    "v": 2, 
    "w": 6, 
    "z": 2, 
    "z^": 1,
    "ng": 6,
    "gn": 8,
}

# Dictionary mapping vowels to position numbers
vowel_to_position = {
    "a": 1, 
    "a~": 3,
    "e": 5, 
    "e^": 4,
    "e~": 2,
    "i": 3, 
    "o": 4, 
    "o^": 1,
    "o~": 3,
    "u": 4, 
    "y": 5, 
    "x": 2, 
    "x^": 1,
    "x~": 5,
}

# Define hand connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)             # Palm connections
]

# Define hand rotations for visualization
HAND_ROTATIONS = {
    1: 20,   # No rotation
    2: 20,  # Rotate 30 degrees clockwise
    3: 20,
    4: 20,
    5: 20,
    6: 20,
    7: 20,
    8: 20 
}

# IPA to target phoneme mapping
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