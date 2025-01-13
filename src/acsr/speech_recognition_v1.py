import os
import subprocess
import whisper
import torch
from moviepy.editor import VideoFileClip
from praatio import textgrid as tgio
import re
import cv2
import numpy as np
import json
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Define the mapping of consonants to hand shapes
consonant_to_handshape = {
    "p": 1, "t": 5, "k": 2, "b": 4, "d": 1, "g": 7, "m": 5, "n": 4,
    "l": 6, "r": 3, "s": 3, "f": 5, "v": 2, "z": 2, "ʃ": 6, "ʒ": 1,
    "ɡ": 7, "ʁ": 3, "j": 8, "w": 6, "ŋ": 8, "ɥ": 4, "ʀ": 3, "y": 8, "c": 2
}

# Define vowel positions relative to the nose (right side of the face/body)
vowel_positions = {
    # Position 1: /a/, /o/, /œ/, /ə/
    "a": (-0.15, 0.1, 0.0),   # Right side of the mouth
    "o": (-0.15, 0.1, 0.0),   # Same as /a/
    "œ": (-0.15, 0.1, 0.0),   # Same as /a/
    "ə": (-0.15, 0.1, 0.0),   # Same as /a/

    # Position 2: /ɛ̃/, /ø/
    "ɛ̃": (-0.2, 0.05, 0.0),   # Right cheek
    "ø": (-0.2, 0.05, 0.0),   # Same as /ɛ̃/

    # Position 3: /i/, /ɔ̃/, /ɑ̃/
    "i": (-0.1, 0.15, 0.0),   # Right corner of the mouth
    "ɔ̃": (-0.1, 0.15, 0.0),   # Same as /i/
    "ɑ̃": (-0.1, 0.15, 0.0),   # Same as /i/

    # Position 4: /u/, /ɛ/, /ɔ/
    "u": (-0.0, 0.2, 0.0),    # Chin (below the mouth)
    "ɛ": (-0.0, 0.2, 0.0),    # Same as /u/
    "ɔ": (-0.0, 0.2, 0.0),    # Same as /u/

    # Position 5: /œ̃/, /y/, /e/
    "œ̃": (-0.0, 0.3, 0.0),    # Throat (below the chin)
    "y": (-0.0, 0.3, 0.0),    # Same as /œ̃/
    "e": (-0.0, 0.3, 0.0),    # Same as /œ̃/
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

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
        "--beam", "200",  # Increase beam size
        "--retry_beam", "400",  # Increase retry beam size
        "--fine_tune",    # Fine-tune the alignment
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


def map_syllable_to_cue(syllable):
    """
    Map a syllable to its corresponding hand shape and hand position.
    Args:
        syllable (str): Syllable in IPA format (e.g., "si", "ne", "ma").
    Returns:
        tuple: (hand_shape, hand_position)
    """
    # Define vowels and consonants
    vowels = set("aeɛioɔuøœəɑ̃ɛ̃ɔ̃œ̃y")
    consonants = set("ptkbdgmnlrsfvzʃʒɡʁjwŋtrɥgʀyc")

    # Check if the syllable is CV, C, or V
    if len(syllable) == 2:  # CV syllable
        consonant, vowel = syllable[0], syllable[1]
        if consonant in consonants and vowel in vowels:
            hand_shape = consonant_to_handshape.get(consonant, 1)  # Default to Hand Shape 1
            hand_position = vowel_positions.get(vowel, (0.15, 0.1, 0.0))  # Default to Position 1
            return hand_shape, hand_position

    elif len(syllable) == 1:  # Single letter (C or V)
        if syllable in consonants:  # Single consonant
            hand_shape = consonant_to_handshape.get(syllable, 1)  # Default to Hand Shape 1
            hand_position = vowel_positions["a"]  # Default to Position 1
            return hand_shape, hand_position
        elif syllable in vowels:  # Single vowel
            hand_shape = 5  # Default to Hand Shape 5
            hand_position = vowel_positions.get(syllable, (0.15, 0.1, 0.0))  # Default to Position 1
            return hand_shape, hand_position

    # Default fallback
    return 1, (0.15, 0.1, 0.0)  # Hand Shape 1, Position 1

def load_hand_landmarks(hand_shape):
    """
    Load hand landmarks from the JSON file for the specified hand shape.
    Args:
        hand_shape (int): Hand shape number (1 to 8).
    Returns:
        dict: Hand landmarks and nose coordinates.
    """
    landmarks_path = f"/scratch2/bsow/Documents/ACSR/data/handshapes/coordinates/handshape_{hand_shape}.json"
    if not os.path.exists(landmarks_path):
        raise FileNotFoundError(f"Hand shape {hand_shape} not found: {landmarks_path}")

    with open(landmarks_path, "r") as f:
        return json.load(f)


def render_hand(frame, hand_landmarks):
    """
    Render the hand landmarks on the frame.
    Args:
        frame: Input video frame.
        hand_landmarks (list): List of hand landmarks.
    """
    # Draw the landmarks
    for landmark in hand_landmarks:
        x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circles for landmarks

    # Draw connections between landmarks
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_x, start_y = int(hand_landmarks[start_idx][0] * frame.shape[1]), int(hand_landmarks[start_idx][1] * frame.shape[0])
        end_x, end_y = int(hand_landmarks[end_idx][0] * frame.shape[1]), int(hand_landmarks[end_idx][1] * frame.shape[0])
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # Blue lines for connections

def choose_reference_finger(hand_landmarks):
    """
    Choose the reference finger (the one with the maximum distance between the base of the palm and the tip).
    Args:
        hand_landmarks (list): List of hand landmarks.
    Returns:
        int: Index of the reference landmark.
    """
    palm_base = hand_landmarks[0]  # Landmark 0 is the base of the palm
    fingertip_indices = [4, 8, 12, 16, 20]  # Indices of the fingertips

    max_distance = 0
    reference_landmark_idx = 8  # Default to index finger tip

    for idx in fingertip_indices:
        fingertip = hand_landmarks[idx]
        distance = ((fingertip[0] - palm_base[0]) ** 2 + (fingertip[1] - palm_base[1]) ** 2 + (fingertip[2] - palm_base[2]) ** 2) ** 0.5
        if distance > max_distance:
            max_distance = distance
            reference_landmark_idx = idx

    return reference_landmark_idx

def adjust_hand_to_nose(hand_landmarks, nose_landmarks):
    """
    Adjust the hand landmarks so that the reference finger aligns with the nose.
    Args:
        hand_landmarks (list): List of hand landmarks.
        nose_landmarks (list): Nose coordinates.
    Returns:
        list: Adjusted hand landmarks.
    """
    # Choose the reference finger
    reference_landmark_idx = choose_reference_finger(hand_landmarks)
    reference_landmark = hand_landmarks[reference_landmark_idx]

    # Calculate the offset to move the reference landmark to the nose
    offset_x = nose_landmarks[0] - reference_landmark[0]
    offset_y = nose_landmarks[1] - reference_landmark[1]
    offset_z = nose_landmarks[2] - reference_landmark[2]

    # Adjust all hand landmarks by the offset
    adjusted_landmarks = []
    for landmark in hand_landmarks:
        adjusted_landmarks.append([
            landmark[0] + offset_x,
            landmark[1] + offset_y,
            landmark[2] + offset_z,
        ])

    return adjusted_landmarks

def adjust_hand_to_vowel(hand_landmarks, nose_landmarks, vowel_position):
    """
    Adjust the hand landmarks to the specified vowel position.
    Args:
        hand_landmarks (list): List of hand landmarks.
        nose_landmarks (list): Nose coordinates.
        vowel_position (tuple): (x, y, z) offset relative to the nose.
    Returns:
        list: Adjusted hand landmarks.
    """
    # Calculate the target position for the reference finger
    target_x = nose_landmarks[0] + vowel_position[0]
    target_y = nose_landmarks[1] + vowel_position[1]
    target_z = nose_landmarks[2] + vowel_position[2]

    # Choose the reference finger
    reference_landmark_idx = choose_reference_finger(hand_landmarks)
    reference_landmark = hand_landmarks[reference_landmark_idx]

    # Calculate the offset to move the reference landmark to the target position
    offset_x = target_x - reference_landmark[0]
    offset_y = target_y - reference_landmark[1]
    offset_z = target_z - reference_landmark[2]

    # Adjust all hand landmarks by the offset
    adjusted_landmarks = []
    for landmark in hand_landmarks:
        adjusted_landmarks.append([
            landmark[0] + offset_x,
            landmark[1] + offset_y,
            landmark[2] + offset_z,
        ])

    return adjusted_landmarks

from moviepy.editor import VideoFileClip, AudioFileClip

def add_audio_to_video(video_path, audio_path, output_path):
    """
    Add the original audio to the rendered video.
    Args:
        video_path (str): Path to the rendered video (without audio).
        audio_path (str): Path to the original audio file.
        output_path (str): Path to save the final video with audio.
    """
    # Load the rendered video
    video_clip = VideoFileClip(video_path)

    # Load the original audio
    audio_clip = AudioFileClip(audio_path)

    # Set the audio of the video to the original audio
    final_clip = video_clip.set_audio(audio_clip)

    # Save the final video with audio
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    # Close the clips
    video_clip.close()
    audio_clip.close()

# Main function (updated rendering logic)
def main():
    # File paths
    video_path = "/scratch2/bsow/Documents/ACSR/data/videos/cinema.mp4"
    audio_path = "/scratch2/bsow/Documents/ACSR/data/transcriptions/output_audio.wav"
    output_dir = "/scratch2/bsow/Documents/ACSR/data/transcriptions"
    rendered_video_path = "/scratch2/bsow/Documents/ACSR/data/videos/cinema_cued.mp4"
    final_video_path = "/scratch2/bsow/Documents/ACSR/data/videos/cinema_cued_with_audio.mp4"

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
        print(f"Alignment saved to: {alignment_output}")

        # Step 4: Construct syllables from the TextGrid file
        syllables = construct_syllables(alignment_output)
        print("Syllables and their intervals:")
        for syllable, interval in syllables.items():
            print(f"{syllable.split('_')[0]}: {interval}")

    # Step 5: Render the video with hand cues
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(rendered_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with MediaPipe FaceMesh to get nose coordinates
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Extract nose landmark (landmark 1)
                nose_landmarks = [
                    face_landmarks.landmark[1].x,
                    face_landmarks.landmark[1].y,
                    face_landmarks.landmark[1].z,
                ]

                # Find the current syllable based on the frame count
                current_time = frame_count / fps
                current_syllable = None
                for syllable, (start, end) in syllables.items():
                    if start <= current_time <= end:
                        current_syllable = syllable.split('_')[0]
                        break

                if current_syllable:
                    # Map the syllable to its hand shape and position
                    hand_shape, hand_position = map_syllable_to_cue(current_syllable)

                    # Load the hand landmarks for the specified hand shape
                    hand_data = load_hand_landmarks(hand_shape)
                    hand_landmarks = hand_data["hand_landmarks"]
                    
                    # Step 2: Align the hand to the nose
                    adjusted_landmarks = adjust_hand_to_nose(hand_landmarks, nose_landmarks)

                    # Step 3: Adjust the hand to the vowel position
                    adjusted_landmarks = adjust_hand_to_vowel(adjusted_landmarks, nose_landmarks, hand_position)

                    # Render the hand on the frame
                    render_hand(frame, adjusted_landmarks)

        # Write the frame to the output video
        out.write(frame)
        frame_count += 1

    # Release the video capture and writer
    cap.release()
    out.release()
    print(f"Rendered video saved to: {rendered_video_path}")

    # Step 6: Add the original audio to the rendered video
    add_audio_to_video(rendered_video_path, audio_path, final_video_path)
    print(f"Final video with audio saved to: {final_video_path}")

if __name__ == "__main__":
    main()