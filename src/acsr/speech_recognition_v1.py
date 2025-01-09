import os
import subprocess
import whisper
import torch
from moviepy.editor import VideoFileClip
from praatio import textgrid as tgio
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

# Step 2: Transcribe the entire audio using Whisper
def transcribe_audio(audio_path, device="cuda"):
    """
    Transcribes the entire audio file using OpenAI's Whisper model.
    Args:
        audio_path (str): Path to the audio file.
        device (str): Device to use for inference ("cuda" for GPU or "cpu" for CPU).
    Returns:
        list: A list of tuples containing (start_time, end_time, text).
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
        segments.append((segment["start"], segment["end"], text))
    
    return segments

# Step 3: Align audio to text using MFA (command-line)
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

# Step 4: Construct syllables from the TextGrid file
def construct_syllables(textgrid_path):
    """
    Construct syllables from the TextGrid file using a manual approach.
    Args:
        textgrid_path (str): Path to the TextGrid file.
    Returns:
        dict: A dictionary mapping syllables to their intervals.
    """
    # Define Cued Speech consonants and vowels
    consonants = "ptkbdgmnlrsfvzʃʒɡʁjwŋtrɥgʀyc"
    vowels = "aeɛioɔuøœəɑ̃ɛ̃ɔ̃œ̃ɑ̃ɔ̃ɑ̃ɔ̃"

    # Load the TextGrid file
    tg = tgio.openTextgrid(textgrid_path, includeEmptyIntervals=False)

    # Get the "phones" tier
    phone_tier = tg.getTier("phones")

    # Construct syllables
    syllables = {}
    i = 0
    syllable_count = 0  # Counter to ensure unique syllable keys

    while i < len(phone_tier.entries):
        start, end, phone = phone_tier.entries[i]

        # If the current phone is a vowel, treat it as a syllable
        if phone in vowels:
            syllable_key = f"{phone}_{syllable_count}"  # Add a unique identifier
            syllables[syllable_key] = (start, end)
            syllable_count += 1
            i += 1

        # If the current phone is a consonant, check the next phone
        elif phone in consonants:
            # Check if there is a next phone
            if i + 1 < len(phone_tier.entries):
                next_start, next_end, next_phone = phone_tier.entries[i + 1]

                # If the next phone is a vowel, combine into a CV syllable
                if next_phone in vowels:
                    syllable = phone + next_phone
                    syllable_key = f"{syllable}_{syllable_count}"  # Add a unique identifier
                    syllables[syllable_key] = (start, next_end)
                    syllable_count += 1
                    i += 2  # Skip the next phone since it's part of the syllable

                # If the next phone is not a vowel, treat the consonant as a standalone syllable
                else:
                    syllable_key = f"{phone}_{syllable_count}"  # Add a unique identifier
                    syllables[syllable_key] = (start, end)
                    syllable_count += 1
                    i += 1

            # If there is no next phone, treat the consonant as a standalone syllable
            else:
                syllable_key = f"{phone}_{syllable_count}"  # Add a unique identifier
                syllables[syllable_key] = (start, end)
                syllable_count += 1
                i += 1

        # If the phone is neither a consonant nor a vowel, skip it
        else:
            print("Skipping phone:", phone)
            i += 1

    return syllables

# Main function
def main():
    # File paths
    video_path = "/scratch2/bsow/Documents/ACSR/data/videos/cinema.mp4"
    audio_path = "/scratch2/bsow/Documents/ACSR/data/transcriptions/output_audio.wav"
    output_dir = "/scratch2/bsow/Documents/ACSR/data/transcriptions"

    # Step 1: Extract audio from the video
    video_duration = extract_audio(video_path, audio_path)

    # Step 2: Transcribe the audio using Whisper
    segments = transcribe_audio(audio_path)
    print("Transcription segments:")
    print(segments)
    # Step 3: Align audio to text using MFA
    for segment in segments:
        start_time, end_time, text = segment
        alignment_output = align_audio_to_text(audio_path, text, output_dir)
        #alignment_output = "/scratch2/bsow/Documents/ACSR/data/transcriptions/output_audio.TextGrid"
        print(f"Alignment saved to: {alignment_output}")

        # Step 4: Construct syllables from the TextGrid file
        syllables = construct_syllables(alignment_output)
        print("Syllables and their intervals:")
        for syllable, interval in syllables.items():
            print(f"{syllable.split('_')[0]}: {interval}")

if __name__ == "__main__":
    main()