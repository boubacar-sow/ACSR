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
    "ɡ": 7, "ʁ": 3, "j": 8, "w": 6, "ŋ": 8, "ɥ": 4, "ʀ": 3, "c": 2
}

# Define vowel positions relative to the nose (right side of the face/body)
vowel_positions = {
    # Position 1: /a/, /o/, /œ/, /ə/
    "a": -1, #(-0.15, 0.1, 0.0),   # Right side of the mouth
    "o": -1, #(-0.15, 0.1, 0.0),   # Same as /a/
    "œ": -1, #(-0.15, 0.1, 0.0),   # Same as /a/
    "ə": -1, #(-0.15, 0.1, 0.0),   # Same as /a/

    # Position 2: /ɛ̃/, /ø/
    "ɛ̃": 50,   # Right cheek
    "ø": 50,   # Same as /ɛ̃/

    # Position 3: /i/, /ɔ̃/, /ɑ̃/
    "i": 57,   # Right corner of the mouth, mediapipe 118
    "ɔ̃": 57,   # Same as /i/
    "ɑ̃": 57,   # Same as /i/

    # Position 4: /u/, /ɛ/, /ɔ/
    "u": 175,    # Chin (below the mouth)
    "ɛ": 175,    # Same as /u/
    "ɔ": 175,    # Same as /u/

    # Position 5: /œ̃/, /y/, /e/
    "œ̃": -2, #(-0.0, 0.3, 0.0),    # Throat (below the chin)
    "y": -2, #(-0.0, 0.3, 0.0),    # Same as /œ̃/
    "e": -2, #(-0.0, 0.3, 0.0),    # Same as /œ̃/
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
    
    # concatenate the segments
    all_segments = {"start": segments[0][0], "end": segments[-1][1], "text": segments[0][2]}
    for i, segment in enumerate(segments):
        if i > 0:
            all_segments["text"] += " " + segment[2]
    return [list(all_segments.values())]

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
        print(start, end, phone)

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

                # Check if the next phone is a vowel and consecutive in time
                if (next_phone in vowels or next_phone == "y") and abs(end - next_start) < 0.01:  # Allow small time gap
                    syllable = phone + next_phone
                    syllable_key = f"{syllable}_{syllable_count}"  # Add a unique identifier
                    syllables[syllable_key] = (start, next_end)
                    syllable_count += 1
                    i += 2  # Skip the next phone since it's part of the syllable

                # If the next phone is not a vowel or not consecutive, treat the consonant as a standalone syllable
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
    consonants = set("ptkbdgmnlrsfvzʃʒɡʁjwŋtrɥgʀc")

    # Check if the syllable is CV, C, or V
    if len(syllable) == 2 or len(syllable)==3:  # CV syllable or for bɔ̃, jɛ̃, dɑ̃ (encoding issus that makes them 3 characters)
        if len(syllable)==3:
            consonant, vowel = syllable[0], syllable[1:]
        else:
            consonant, vowel = syllable[0], syllable[1]
        if len(vowel)==2:  # for bɔ̃, jɛ̃, dɑ̃ (encoding issus that makes them 3 characters)
            hand_shape = consonant_to_handshape.get(consonant, 8)  # Default to Hand Shape 1
            if vowel[0] == "ɔ":
                hand_position = vowel_positions.get("ɔ̃", 57)  # Default to Position 3
            elif vowel[0] == "ɛ":
                hand_position = vowel_positions.get("ɛ̃", 50)  # Default to Position 2
            elif vowel[0] == "ɑ":
                hand_position = vowel_positions.get("ɑ̃", 57)
            elif vowel[0] == "œ":
                hand_position = vowel_positions.get("œ̃", -2)
            try:
                return hand_shape, hand_position
            except:
                print("Syllable error: ", syllable)
                raise ValueError("Error")
        
        if consonant in consonants and vowel in vowels:
            hand_shape = consonant_to_handshape.get(consonant, 8)  # Default to Hand Shape 1
            hand_position = vowel_positions.get(vowel, -1)  # Default to Position 1
            return hand_shape, hand_position

    elif len(syllable) == 1:  # Single letter (C or V)
        if syllable in consonants:  # Single consonant
            hand_shape = consonant_to_handshape.get(syllable, 8)  # Default to Hand Shape 1
            hand_position = vowel_positions["a"]  # Default to Position 1
            return hand_shape, hand_position
        elif syllable in vowels:  # Single vowel
            hand_shape = 5  # Default to Hand Shape 5
            hand_position = vowel_positions.get(syllable, -1)  # Default to Position 1
            return hand_shape, hand_position

    # Default fallback
    print(f"Mapping syllable {syllable}, len syllabe = {len(syllable)}, syllabe[0]={syllable[0]}, syllabe[1]={syllable[1]}, syllabe[2]={syllable[2]},  to Hand Shape 1 and Position 1")
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

def calculate_face_scale(face_landmarks, reference_face_bbox):
    """
    Calculate the scale factor based on the face bounding box.
    Args:
        face_landmarks: MediaPipe face landmarks.
    Returns:
        float: Scale factor for the hand.
    """
    # Extract all x and y coordinates of the face landmarks
    x_coords = [landmark.x for landmark in face_landmarks.landmark]
    y_coords = [landmark.y for landmark in face_landmarks.landmark]

    # Calculate the width and height of the face bounding box
    face_width = max(x_coords) - min(x_coords)
    face_height = max(y_coords) - min(y_coords)

    # Use the average of width and height as the face size
    face_size = (face_width + face_height) / 2

    # Define a reference face size (empirically determined for a "normal" face size)
    ref_face_width =  reference_face_bbox["x_max"] - reference_face_bbox["x_min"]
    ref_face_height = reference_face_bbox["y_max"] - reference_face_bbox["y_min"]
    reference_face_size = (ref_face_width + ref_face_height) / 2

    # Calculate the scale factor
    scale_factor = face_size / reference_face_size
    return scale_factor


def adjust_hand_to_vowel(handshape_index, hand_landmarks, face_landmarks, vowel_position, face_bbox, scale_factor=1.0):
    """
    Adjust the hand landmarks to the specified vowel position.
    Args:
        hand_landmarks (list): List of hand landmarks.
        nose_landmarks (list): Nose coordinates.
        vowel_position (tuple): (x, y, z) offset relative to the nose.
        face_bbox (dict): Face bounding box with left, right, top, bottom coordinates.
        scale_factor (float): Scale factor for the hand.
    Returns:
        list: Adjusted hand landmarks.
    """
    # Calculate the target position for the reference finger
    if vowel_position == -1:
        target_x, target_y, target_z = face_landmarks.landmark[57].x - 0.06, face_landmarks.landmark[57].y, face_landmarks.landmark[57].z 
    elif vowel_position == -2:
        target_x, target_y, target_z = face_landmarks.landmark[118].x, face_landmarks.landmark[118].y + 0.04, face_landmarks.landmark[118].z
    else:
        target_x =  face_landmarks.landmark[vowel_position].x 
        target_y = face_landmarks.landmark[vowel_position].y
        target_z = face_landmarks.landmark[vowel_position].z

    # Choose the reference finger
    reference_landmark_idx = choose_reference_finger(handshape_index)
    reference_landmark = hand_landmarks[reference_landmark_idx]

    # Calculate the offset to move the reference landmark to the target position
    offset_x = target_x - reference_landmark[0]*scale_factor*0.45
    offset_y = target_y - reference_landmark[1]*scale_factor*0.45
    offset_z = target_z - reference_landmark[2]*scale_factor*0.45

    # Adjust all hand landmarks by the offset
    adjusted_landmarks = []
    for landmark in hand_landmarks:
        adjusted_landmarks.append([
            landmark[0]*scale_factor*0.45 + offset_x,
            landmark[1]*scale_factor*0.45 + offset_y,
            landmark[2]*scale_factor*0.45 + offset_z,
        ])

    return adjusted_landmarks

def render_hand(frame, hand_landmarks, scale_factor=1.0):
    """
    Render the hand landmarks on the frame.
    Args:
        frame: Input video frame.
        hand_landmarks (list): List of hand landmarks.
        scale_factor (float): Scale factor for the hand landmarks.
    """
    # Scale the hand landmarks
    scaled_landmarks = []
    for landmark in hand_landmarks:
        scaled_landmarks.append([
            landmark[0] ,#* scale_factor,
            landmark[1] ,#* scale_factor,
            landmark[2] ,#* scale_factor,
        ])

    # Draw the landmarks
    for landmark in scaled_landmarks:
        x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circles for landmarks

    # Draw connections between landmarks
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_x, start_y = int(scaled_landmarks[start_idx][0] * frame.shape[1]), int(scaled_landmarks[start_idx][1] * frame.shape[0])
        end_x, end_y = int(scaled_landmarks[end_idx][0] * frame.shape[1]), int(scaled_landmarks[end_idx][1] * frame.shape[0])
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # Blue lines for connections

def choose_reference_finger(consonant_index):
    """
    Choose the reference finger (the one with the maximum distance between the base of the palm and the tip).
    Args:
        hand_landmarks (list): List of hand landmarks.
    Returns:
        int: Index of the reference landmark.
    """
    
    if consonant_index == 1 or consonant_index == 6:
        return 8
    else:
        return 12

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
    final_clip.write_videofile(output_path, codec="libx265", audio_codec="aac")

    # Close the clips
    video_clip.close()
    audio_clip.close()


# Main function (updated rendering logic)
def main():
    # File paths
    video_path = "/scratch2/bsow/Documents/ACSR/data/videos/cuedspeech2.mp4"
    audio_path = "/scratch2/bsow/Documents/ACSR/data/transcriptions/cuedspeech2.wav"
    output_dir = "/scratch2/bsow/Documents/ACSR/data/transcriptions"
    rendered_video_path = "/scratch2/bsow/Documents/ACSR/data/videos/cuedspeech2_lpc.mp4"
    final_video_path = "/scratch2/bsow/Documents/ACSR/data/videos/cuedspeech2_with_audio.mp4"

    # Step 1: Extract audio from the video
    video_duration = extract_audio(video_path, audio_path)

    # Step 2: Transcribe the audio using Whisper
    segments = transcribe_audio(audio_path)
    print("Transcription segments:")
    print(segments)

    all_syllables = {}

    # Step 3: Align audio to text using MFA
    max = 0
    for segment in segments:
        start_time, end_time, text = segment
        alignment_output = align_audio_to_text(audio_path, text, output_dir)
        print(f"Alignment saved to: {alignment_output}")

        # Step 4: Construct syllables from the TextGrid file
        syllables = construct_syllables(alignment_output)
        print("Syllables and their intervals:")
        for i, (syllable, interval) in enumerate(syllables.items()):
            #if i == 0:
            #    syllable[0]= max
            print(f"{syllable.split('_')[0]}: {interval}")
            all_syllables[syllable] = interval
            max = interval[1]
        

    # Step 5: Render the video with hand cues
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use H.264 codec here
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
                for syllable, (start, end) in all_syllables.items():
                    if start <= current_time <= end:
                        current_syllable = syllable.split('_')[0]
                        break

                if current_syllable:
                    # Map the syllable to its hand shape and position
                    hand_shape, hand_position = map_syllable_to_cue(current_syllable)

                    # Load the hand landmarks for the specified hand shape
                    hand_and_ref_data = load_hand_landmarks(hand_shape)
                    hand_landmarks = hand_and_ref_data["hand_landmarks"]
                    reference_face_bbox = hand_and_ref_data["face_bbox"]

                    # Calculate the face scale
                    scale_factor = calculate_face_scale(face_landmarks, reference_face_bbox)
                    
                    # Step 3: Adjust the hand to the vowel position
                    adjusted_landmarks = adjust_hand_to_vowel(hand_shape, hand_landmarks, face_landmarks, hand_position, scale_factor)

                    # Render the hand on the frame with scaling
                    render_hand(frame, adjusted_landmarks, scale_factor)

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