import os
import subprocess
from moviepy.editor import VideoFileClip
import cv2
from praatio import textgrid as tgio
import whisper
import torch

# Paths
video_dir = "/scratch2/bsow/Documents/ACSR/data/training_videos/videos"
audio_dir = "/scratch2/bsow/Documents/ACSR/data/training_videos/audio"
textgrid_dir = "/scratch2/bsow/Documents/ACSR/data/training_videos/textgrids"
annotation_dir = "/scratch2/bsow/Documents/ACSR/data/training_videos/annotations"
frames_to_predict_dir = "/scratch2/bsow/Documents/ACSR/data/training_videos/frames_to_predict"
lpc_dir = "/scratch2/bsow/Documents/ACSR/data/training_videos/lpc"

# Create directories if they don't exist
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(textgrid_dir, exist_ok=True)
os.makedirs(annotation_dir, exist_ok=True)
os.makedirs(frames_to_predict_dir, exist_ok=True)
os.makedirs(lpc_dir, exist_ok=True)

# Load Whisper model (use GPU if available)
whisper_model = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")

# Function to extract audio from video
def extract_audio(video_path, audio_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    print(f"Audio extracted and saved to: {audio_path}")

# Function to transcribe audio using Whisper
def transcribe_audio_with_whisper(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Transcribe the audio
    result = whisper_model.transcribe(audio_path, language="fr")
    transcription = result["text"]
    print(f"Transcription: {transcription}")
    return transcription

# Function to align audio using MFA
def align_audio_with_mfa(audio_path, text, output_dir):
    mfa_input_dir = os.path.join(output_dir, "mfa_input")
    os.makedirs(mfa_input_dir, exist_ok=True)

    # Copy audio and text to MFA input directory
    audio_filename = os.path.basename(audio_path)
    mfa_audio_path = os.path.join(mfa_input_dir, audio_filename)
    os.system(f"cp {audio_path} {mfa_audio_path}")

    # Save the text to a corresponding .lab file
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
        "--beam", "200",  # Increase beam size
        "--retry_beam", "400",  # Increase retry beam size
        "--fine_tune",    # Fine-tune the alignment
    ]
    subprocess.run(command, check=True)

    # Return the path to the TextGrid file
    return os.path.join(output_dir, f"{os.path.splitext(audio_filename)[0]}.TextGrid")

# Function to construct syllables from the TextGrid file
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

# Function to save syllables to an .lpc file
def save_syllables_to_lpc(syllables, video_name):
    """
    Save syllables to an .lpc file.
    Args:
        syllables (dict): Dictionary of syllables and their intervals.
        video_name (str): Name of the video (e.g., "sent_01").
    """
    lpc_path = os.path.join(lpc_dir, f"{video_name}.lpc")
    with open(lpc_path, "w") as f:
        for syllable_key, (start, end) in syllables.items():
            f.write(f"{syllable_key} {start} {end}\n")
    print(f"Syllables saved to: {lpc_path}")

# Function to extract frames from the middle of each syllable interval
def extract_frames_from_syllables(video_path, syllables, output_dir):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # List to store frame indices
    frame_indices = []

    # Iterate through syllable intervals
    for syllable_key, (start_time, end_time) in syllables.items():
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        num_frames = end_frame - start_frame

        # Determine the frames to extract
        if num_frames >= 3:
            frame_indices.append(start_frame + num_frames // 2)
        elif num_frames == 2:
            frame_indices.append(start_frame + 1)
        elif num_frames == 1:
            frame_indices.append(start_frame)

    # Save frame indices to a text file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_to_predict_path = os.path.join(frames_to_predict_dir, f"{video_name}.txt")
    with open(frames_to_predict_path, "w") as f:
        for frame_idx in frame_indices:
            f.write(f"{frame_idx}\n")
    print(f"Frame indices saved to: {frames_to_predict_path}")

    # Extract and save the frames
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = f"frame_{frame_idx}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"Frame saved: {frame_path}")

    cap.release()

# Main function
def main():
    # Iterate through all videos from sent_01.mp4 to sent_30.mp4
    for i in range(1, 16):
        video_filename = f"sent_{i:02d}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        audio_path = os.path.join(audio_dir, f"sent_{i:02d}.wav")
        textgrid_path = os.path.join(textgrid_dir, f"sent_{i:02d}.TextGrid")
        annotation_video_dir = os.path.join(annotation_dir, f"sent_{i:02d}")
        os.makedirs(annotation_video_dir, exist_ok=True)

        # Step 1: Extract audio
        extract_audio(video_path, audio_path)

        # Step 2: Transcribe audio using Whisper
        transcription = transcribe_audio_with_whisper(audio_path)

        # Step 3: Align audio with MFA using the Whisper transcription
        align_audio_with_mfa(audio_path, transcription, textgrid_dir)

        # Step 4: Construct syllables from the TextGrid file
        syllables = construct_syllables(textgrid_path)

        # Step 5: Save syllables to an .lpc file
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_syllables_to_lpc(syllables, video_name)

        # Step 6: Extract frames from syllable intervals and save frame indices
        extract_frames_from_syllables(video_path, syllables, annotation_video_dir)

if __name__ == "__main__":
    main()