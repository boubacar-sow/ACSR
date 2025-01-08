import os
import cv2
import whisper
import torch
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from epitran.backoff import Backoff
import re
import mediapipe as mp

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

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

# Step 5: Extract landmarks using MediaPipe
def extract_landmarks(frame):
    """
    Extract head and hand landmarks from a video frame using MediaPipe Holistic.
    Args:
        frame: Input video frame.
    Returns:
        dict: Landmarks for face, right hand, and left hand.
    """
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    # Extract landmarks
    landmarks = {
        "face": results.face_landmarks,
        "right_hand": results.right_hand_landmarks,
        "left_hand": results.left_hand_landmarks,
    }
    return landmarks

# Step 6: Build syllable-to-gesture mappings
def build_syllable_mappings(video_path, syllable_annotations):
    """
    Build syllable-to-gesture mappings by extracting hand coordinates during annotated frames.
    Args:
        video_path: Path to the video file.
        syllable_annotations: List of syllable annotations (start_frame, end_frame, syllable).
    Returns:
        dict: Syllable-to-gesture mappings.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    syllable_mappings = {}

    for annotation in syllable_annotations:
        syllable = annotation["syllable"]
        start_frame = annotation["start_frame"]
        end_frame = annotation["end_frame"]

        # Set the video to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Extract hand coordinates for the syllable
        hand_coordinates = []
        for _ in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = extract_landmarks(frame)
            if landmarks["right_hand"]:
                # Extract x, y, z coordinates from the right hand landmarks
                hand_coords = []
                for landmark in landmarks["right_hand"].landmark:
                    hand_coords.append([landmark.x, landmark.y, landmark.z])
                hand_coordinates.append(hand_coords)

        # Map the syllable to the hand coordinates
        if hand_coordinates:
            syllable_mappings[syllable] = hand_coordinates

    cap.release()
    return syllable_mappings

# Step 7: Render gestures on the video
def render_gestures(video_path, syllable_mappings, syllable_annotations, output_video_path):
    """
    Render gestures on the video by overlaying hand positions on the head.
    Args:
        video_path: Path to the input video.
        syllable_mappings: Syllable-to-gesture mappings.
        syllable_annotations: List of syllable annotations (start_frame, end_frame, syllable).
        output_video_path: Path to save the output video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Define connections between hand landmarks (based on MediaPipe hand connections)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract landmarks
        landmarks = extract_landmarks(frame)

        # Create a blank frame (black background)
        blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Reconstruct the face using MediaPipe landmarks
        if landmarks["face"]:
            for landmark in landmarks["face"].landmark:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(blank_frame, (x, y), 2, (255, 255, 255), -1)  # Draw white dots for face landmarks

        # Find the current syllable based on the frame count
        current_syllable = None
        current_annotation = None
        for annotation in syllable_annotations:
            if annotation["start_frame"] <= frame_count <= annotation["end_frame"]:
                current_syllable = annotation["syllable"]
                current_annotation = annotation
                break

        # Overlay hand gestures based on the current syllable
        if current_syllable and current_syllable in syllable_mappings:
            hand_coordinates = syllable_mappings[current_syllable]
            if landmarks["face"] and hand_coordinates:
                # Calculate the index for the current frame within the syllable's range
                frame_index = frame_count - current_annotation["start_frame"]
                if 0 <= frame_index < len(hand_coordinates):
                    # Draw hand landmarks and connections
                    for i, coord in enumerate(hand_coordinates[frame_index]):
                        x = int(coord[0] * frame_width)
                        y = int(coord[1] * frame_height)
                        cv2.circle(blank_frame, (x, y), 5, (0, 255, 0), -1)  # Draw green dots for hand landmarks

                    # Draw connections between hand landmarks
                    for connection in HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        start_x = int(hand_coordinates[frame_index][start_idx][0] * frame_width)
                        start_y = int(hand_coordinates[frame_index][start_idx][1] * frame_height)
                        end_x = int(hand_coordinates[frame_index][end_idx][0] * frame_width)
                        end_y = int(hand_coordinates[frame_index][end_idx][1] * frame_height)
                        cv2.line(blank_frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

                    # Display the syllable text at the start of the cue
                    if frame_index == 0:
                        cv2.putText(blank_frame, current_syllable, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the frame to the output video
        out.write(blank_frame)
        frame_count += 1

    # Add a final frame with the complete transcribed text
    final_text = " ".join([annotation["syllable"] for annotation in syllable_annotations])
    final_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    cv2.putText(final_frame, final_text, (50, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for _ in range(int(fps * 2)):  # Display the final text for 2 seconds
        out.write(final_frame)

    cap.release()
    out.release()

# Step 8: Combine video and audio
def combine_video_audio(video_path, audio_path, output_path):
    """
    Combine a video with no audio and an audio file into a single video file.
    Args:
        video_path: Path to the video file (without audio).
        audio_path: Path to the audio file.
        output_path: Path to save the final video file.
    """
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec="libx264")

# Main function
def main():
    # File paths
    video_path = "/scratch2/bsow/Documents/ACSR/data/training_videos/sent_01.mp4"  # Replace with your video file path
    audio_path = "/scratch2/bsow/Documents/ACSR/data/transcriptions/output_audio.wav"  # Temporary audio file
    output_video_path = "/scratch2/bsow/Documents/ACSR/data/transcriptions/output_video.mp4"  # Output video file
    final_output_path = "/scratch2/bsow/Documents/ACSR/data/transcriptions/final_output.mp4"  # Final video with audio

    # Step 1: Extract audio from the video
    video_duration = extract_audio(video_path, audio_path)

    # Step 2: Transcribe the entire audio
    device = "cuda"  # Set to "cuda" for GPU or "cpu" for CPU
    segments = transcribe_audio(audio_path, device=device)

    # Step 3: Build syllable-to-gesture mappings
    syllable_annotations = [
        {"syllable": "y", "start_frame": 13, "end_frame": 20},
        {"syllable": "go", "start_frame": 24, "end_frame": 38},
        {"syllable": "vi", "start_frame": 42, "end_frame": 45},
        {"syllable": "vɛ", "start_frame": 47, "end_frame": 55},
        {"syllable": "dɑ̃", "start_frame": 60, "end_frame": 67},
        {"syllable": "y", "start_frame": 72, "end_frame": 80},
        {"syllable": "n", "start_frame": 82, "end_frame": 90}, 
        {"syllable": "ka", "start_frame": 93, "end_frame": 100},
        {"syllable": "ba", "start_frame": 101, "end_frame": 104},
        {"syllable": "n", "start_frame": 105, "end_frame": 124},
        {"syllable": "ki", "start_frame": 128, "end_frame": 134},
        {"syllable": "la", "start_frame": 138, "end_frame": 140},
        {"syllable": "vɛ", "start_frame": 146, "end_frame": 152},
        {"syllable": "lu", "start_frame": 154, "end_frame": 160},
        {"syllable": "mɛ", "start_frame": 164, "end_frame": 172},
        {"syllable": "m", "start_frame": 174, "end_frame": 182},
        {"syllable": "kɔ̃", "start_frame": 193, "end_frame": 202},
        {"syllable": "s", "start_frame": 207, "end_frame": 211},
        {"syllable": "t", "start_frame": 212, "end_frame": 217},
        {"syllable": "ru", "start_frame": 218, "end_frame": 222},
        {"syllable": "t", "start_frame": 224, "end_frame": 227}
    ]
    syllable_mappings = build_syllable_mappings(video_path, syllable_annotations)

    # Step 4: Render gestures on the video
    render_gestures(video_path, syllable_mappings, syllable_annotations, output_video_path)

    # Step 5: Combine video and audio
    combine_video_audio(output_video_path, audio_path, final_output_path)

    # Clean up temporary audio file
    os.remove(audio_path)
    print("Temporary audio file removed.")

if __name__ == "__main__":
    main()