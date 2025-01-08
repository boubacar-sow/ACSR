import os
import cv2
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
consonants = "ptkbdgmnlrsfvzʃʒʁjwŋtrɥʀ"
vowels = "aeɛioɔuyøœəɑ̃ɛ̃ɔ̃œ̃ɔ̃ɑ̃"

# Regex pattern for syllabification
syllable_pattern = re.compile(
    f"[{consonants}]?[{vowels}]|[{consonants}]", re.IGNORECASE
)

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
    words = sentence.split()
    syllabified_sentence = []
    for word in words:
        syllabified_sentence.append(syllabify_word(word))
    return " ".join(syllabified_sentence)

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

# Step 5: Save video with overlaid text
def save_video_with_text(video_path, segments, output_video_path):
    """
    Saves the video with overlaid text (original, IPA, and syllabified) without displaying it.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    current_text = ""
    current_ipa = ""
    current_syllabified = ""
    segment_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current time in the video
        current_time = frame_count / fps

        # Check if the current time falls within the next segment
        if segment_index < len(segments):
            start_time, end_time, text, ipa_text, syllabified_text = segments[segment_index]
            if start_time <= current_time <= end_time:
                current_text = text
                current_ipa = ipa_text
                current_syllabified = syllabified_text
            elif current_time > end_time:
                segment_index += 1  # Move to the next segment

        # Overlay the text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        font_thickness = 2
        text_position = (50, 50)  # Top-left corner
        line_spacing = 40  # Space between lines

        # Overlay original text
        frame = cv2.putText(frame, f"Text: {current_text}", text_position, font, font_scale, font_color, font_thickness)

        # Overlay IPA text
        frame = cv2.putText(frame, f"IPA: {current_ipa}", (text_position[0], text_position[1] + line_spacing), font, font_scale, font_color, font_thickness)

        # Overlay syllabified text
        frame = cv2.putText(frame, f"Syllabified: {current_syllabified}", (text_position[0], text_position[1] + 2 * line_spacing), font, font_scale, font_color, font_thickness)

        # Write the frame to the output video
        out.write(frame)

        frame_count += 1

    # Release the video objects
    cap.release()
    out.release()
    print(f"Processed video saved to: {output_video_path}")

# Main function
def main():
    # File paths
    video_path = "/scratch2/bsow/Documents/ACSR/data/training_videos/sent_01.mp4"  # Replace with your video file path
    audio_path = "/scratch2/bsow/Documents/ACSR/data/transcriptions/output_audio.wav"  # Temporary audio file
    output_video_path = "/scratch2/bsow/Documents/ACSR/data/transcriptions/output_video.mp4"  # Output video file

    # Step 1: Extract audio from the video
    extract_audio(video_path, audio_path)

    # Step 2: Transcribe the entire audio
    device = "cuda"  # Set to "cuda" for GPU or "cpu" for CPU
    segments = transcribe_audio(audio_path, device=device)

    # Step 3: Save video with overlaid text
    save_video_with_text(video_path, segments, output_video_path)

    # Clean up temporary audio file
    os.remove(audio_path)
    print("Temporary audio file removed.")

if __name__ == "__main__":
    main()