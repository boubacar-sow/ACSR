import os
import subprocess
import whisper
import torch
from moviepy.editor import VideoFileClip
from epitran.backoff import Backoff
import re

# Step 1: Extract audio from the video
def extract_audio(video_path, audio_path):
    """
    Extracts audio from a video file and saves it as a WAV file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    print(f"Audio extracted and saved to: {audio_path}")
    return video_clip.duration  # Return the duration of the video

# Step 2: Convert text to IPA
def text_to_ipa(text, language="fra-Latn"):
    """
    Convert a text sentence into its IPA representation.
    Args:
        text (str): Input text.
        language (str): Language code for IPA conversion (e.g., "fra-Latn" for French).
    Returns:
        str: IPA representation of the text.
    """
    backoff = Backoff([language])
    ipa_text = backoff.transliterate(text)
    return ipa_text

# Step 3: Syllabify IPA text
# Define Cued Speech consonants (hand shapes) and vowels (mouth shapes)
consonants = "ptkbdgmnlrsfvzʃʒɡʁjwŋtrɥʀ"
vowels = "aeɛioɔuyøœəɑ̃ɛ̃ɔ̃œ̃ɑ̃ɔ̃ɑ̃ɔ̃"

# Regex pattern for syllabification
syllable_pattern = re.compile(
    f"[{consonants}]?[{vowels}]|[{consonants}]", re.IGNORECASE
)

ipa_to_arpabet = {
    'p': 'P', 't': 'T', 'k': 'K', 'b': 'B', 'd': 'D', 'g': 'G',
    'm': 'M', 'n': 'N', 'l': 'L', 'r': 'R', 's': 'S', 'f': 'F',
    'v': 'V', 'z': 'Z', 'ʃ': 'SH', 'ʒ': 'ZH', 'ɡ': 'G', 'ʁ': 'RH',
    'j': 'Y', 'w': 'W', 'ŋ': 'NG', 'tɾ': 'TR', 'ɥ': 'HW', 'ʀ': 'RR',
    'a': 'AA', 'e': 'EY', 'ɛ': 'EH', 'i': 'IY', 'o': 'OW', 'ɔ': 'AO',
    'u': 'UW', 'y': 'UY', 'ø': 'OE', 'œ': 'EU', 'ə': 'AH', 'ɑ̃': 'AN',
    'ɛ̃': 'EN', 'ɔ̃': 'ON', 'œ̃': 'UN', 'ɡ': 'g', 'ʁ': 'r',
}

def syllabify_word(word):
    """
    Syllabify a single word based on the allowed patterns: CV, V, C.
    """
    syllables = syllable_pattern.findall(word)
    return " ".join(syllables)

def syllabify_sentence(sentence):
    """
    Syllabify an entire sentence.
    """
    sentence = sentence.lower()
    sentence = text_to_ipa(sentence)
    words = sentence.split()
    syllabified_sentence = []
    for word in words:
        syllabified_sentence.append(syllabify_word(word))
    return " ".join(syllabified_sentence)

def ipa_to_arpabet(ipa_text):
    """
    Convert IPA text to ARPAbet phonemes.
    Args:
        ipa_text (str): IPA text.
    Returns:
        str: ARPAbet phonemes.
    """
    arpabet_phonemes = [ipa_to_arpabet[char] for char in ipa_text.split()]
    return " ".join(arpabet_phonemes)

# Step 4: Transcribe the entire audio using Whisper
def transcribe_audio(audio_path, device="cuda"):
    """
    Transcribes the entire audio file using OpenAI's Whisper model.
    Args:
        audio_path (str): Path to the audio file.
        device (str): Device to use for inference ("cuda" for GPU or "cpu" for CPU).
    Returns:
        list: A list of tuples containing (start_time, end_time, text, ipa_text, syllabified_text).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Check if the specified device is available
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    # Load the Whisper model on the specified device
    model = whisper.load_model("medium", device=device)  # Use "medium" or "large" for better accuracy

    # Transcribe the entire audio file
    result = model.transcribe(audio_path, language="fr")
    print("Audio transcription completed.")

    # Extract segments from the result
    segments = []
    for segment in result["segments"]:
        text = segment["text"]
        ipa_text = text_to_ipa(text)  # Convert text to IPA
        syllabified_text = syllabify_sentence(ipa_text)  # Syllabify IPA text
        segments.append((segment["start"], segment["end"], text, ipa_text, syllabified_text))
    
    return segments

# Step 5: Align audio to text using MFA (command-line)
def align_audio_to_text(audio_path, text, output_dir):
    """
    Align audio to text using Montreal Forced Aligner (MFA).
    Args:
        audio_path (str): Path to the audio file.
        text (str): Original non-syllabified text.
        output_dir (str): Directory to save alignment results.
    Returns:
        str: Path to the alignment output (TextGrid file).
    """
    # Create a temporary directory for MFA input
    mfa_input_dir = os.path.join(output_dir, "mfa_input")
    os.makedirs(mfa_input_dir, exist_ok=True)

    # Copy the audio file to the MFA input directory
    audio_filename = os.path.basename(audio_path)
    mfa_audio_path = os.path.join(mfa_input_dir, audio_filename)
    os.system(f"cp {audio_path} {mfa_audio_path}")

    # Save the original text to a corresponding .lab file
    text_filename = os.path.splitext(audio_filename)[0] + ".lab"
    text_path = os.path.join(mfa_input_dir, text_filename)
    with open(text_path, "w") as f:
        f.write(text)

    # Run MFA alignment
    command = [
        "mfa", "align",
        mfa_input_dir,  # Directory containing audio and text files
        "french_mfa",   # Dictionary name
        "french_mfa",   # Acoustic model name
        output_dir,     # Output directory for alignment results
        "--clean",      # Clean the output directory before running
        "--beam", "100",  # Increase beam size
        "--retry_beam", "400",  # Increase retry beam size
    ]
    subprocess.run(command, check=True)

    # Return the path to the TextGrid file
    return os.path.join(output_dir, f"{os.path.splitext(audio_filename)[0]}.TextGrid")


# Main function
def main():
    # File paths
    video_path = "/scratch2/bsow/Documents/ACSR/data/training_videos/sent_01.mp4"
    audio_path = "/scratch2/bsow/Documents/ACSR/data/transcriptions/output_audio.wav"
    output_dir = "/scratch2/bsow/Documents/ACSR/data/transcriptions"

    # Step 1: Extract audio from the video
    video_duration = extract_audio(video_path, audio_path)

    # Step 2: Transcribe the audio using Whisper
    segments = transcribe_audio(audio_path)

    # Step 3: Align audio to text using MFA
    for segment in segments:
        start_time, end_time, text, ipa_text, syllabified_text = segment
        alignment_output = align_audio_to_text(audio_path, text, output_dir)
        print(f"Alignment saved to: {alignment_output}")

if __name__ == "__main__":
    main()