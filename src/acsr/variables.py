# variables.py

# Define the mapping of consonants to hand shapes
consonant_to_handshape = {
    "p": 1, "t": 5, "k": 2, "b": 4, "d": 1, "g": 7, "m": 5, "n": 4,
    "l": 6, "r": 3, "s": 3, "f": 5, "v": 2, "z": 2, "ʃ": 6, "ʒ": 1,
    "ɡ": 7, "ʁ": 3, "j": 8, "w": 6, "ŋ": 8, "ɥ": 4, "ʀ": 3, "y": 8, "c": 2
}

# Define vowel positions relative to the nose (right side of the face/body)
vowel_positions = {
    # Position -1: /a/, /o/, /œ/, /ə/
    "a": -1,
    "o": -1,
    "œ": -1,
    "ə": -1,

    # Position 50: /ɛ̃/, /ø/
    "ɛ̃": 50,
    "ø": 50,

    # Position 57: /i/, /ɔ̃/, /ɑ̃/
    "i": 57,
    "ɔ̃": 57,
    "ɑ̃": 57,

    # Position 175: /u/, /ɛ/, /ɔ/
    "u": 175,
    "ɛ": 175,
    "ɔ": 175,

    # Position -2: /œ̃/, /y/, /e/
    "œ̃": -2,
    "y": -2,
    "e": -2,
}

# Define hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)             # Palm connections
]
