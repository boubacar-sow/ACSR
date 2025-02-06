import json
import logging
import os
import subprocess
from bisect import bisect_left

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import whisper
from moviepy.editor import AudioFileClip, VideoFileClip
from praatio import textgrid as tgio

from speech_recognition_v1 import (calculate_face_scale, 
                                   map_syllable_to_cue)
from variables import HAND_CONNECTIONS, consonant_to_handshape, vowel_positions, HAND_ROTATIONS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CONFIG = {
    "video_path": "/scratch2/bsow/Documents/ACSR/data/videos/cuedspeech2.mp4",
    "output_dir": "/scratch2/bsow/Documents/ACSR/data/transcriptions",
    "handshapes_dir": "/scratch2/bsow/Documents/ACSR/data/handshapes/coordinates",
    "language": "french",
    "reference_face_size": 0.3,  # Normalized reference face size
    "hand_scale_factor": 0.75,
    "mfa_args": ["--beam", "200", "--retry_beam", "400", "--fine_tune"],
    "video_codec": "libx265",
    "audio_codec": "aac"
}

# Initialize MediaPipe once
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

class CuedSpeechProcessor:
    def __init__(self, config):
        self.config = config
        self.syllable_map = []
        self.current_video_frame = None
        self._validate_paths()

    def _validate_paths(self):
        """Ensure required directories and files exist"""
        if not os.path.exists(self.config["video_path"]):
            raise FileNotFoundError(f"Video file not found: {self.config['video_path']}")
        
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def process_video(self):
        """Main processing pipeline"""
        try:
            # Pipeline steps
            audio_path = self._extract_audio()
            transcription = self._transcribe_audio(audio_path)
            self._align_and_build_syllables(audio_path, transcription)
            rendered_video = self._render_video()
            final_output = self._add_audio(rendered_video, audio_path, os.path.join(
                                        self.config["output_dir"],
                                        f"final_{os.path.basename(rendered_video)}"
                                    ))
            
            logging.info(f"Processing complete. Final output: {final_output}")
            return final_output
            
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            raise

    def _extract_audio(self):
        """Extract audio from video file"""
        audio_path = os.path.join(self.config["output_dir"], "audio.wav")
        
        with VideoFileClip(self.config["video_path"]) as video:
            video.audio.write_audiofile(audio_path, codec="pcm_s16le")
            logging.info(f"Audio extracted to {audio_path}")
            return audio_path

    def _transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("medium", device=device)
        result = model.transcribe(audio_path, language=self.config["language"])
        
        logging.info("Audio transcription completed")
        return result["text"]

    def _align_and_build_syllables(self, audio_path, text):
        """Align text and build syllable timeline"""
        textgrid_path =  "/scratch2/bsow/Documents/ACSR/data/transcriptions/cuedspeech2.TextGrid" # self._run_mfa_alignment(audio_path, text)
        self.syllable_map = self._parse_textgrid(textgrid_path)
        assert len(self.syllable_map) > 0, "Syllable map is empty"

        # Sort by start time for efficient lookup
        self.syllable_map.sort(key=lambda x: x[1])
        print(self.syllable_map)
        self.syllable_times = [item[1] for item in self.syllable_map]

    def _run_mfa_alignment(self, audio_path, text):
        """Run Montreal Forced Aligner"""
        # Create a temporary directory for MFA input
        mfa_input_dir = os.path.join(self.config['output_dir'], "mfa_input")
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
        # Build MFA command
        cmd = [
            "mfa", "align",
            mfa_input_dir,
            f"{self.config['language']}_mfa",
            f"{self.config['language']}_mfa",
            self.config["output_dir"],
            "--clean"
        ] + self.config["mfa_args"]

        subprocess.run(cmd, check=True)
        return os.path.join(self.config["output_dir"], f"{os.path.splitext(audio_filename)[0]}.TextGrid")

    def _parse_textgrid(self, textgrid_path):
        """
        Parse TextGrid into syllable timeline using manual syllable construction.
        Args:
            textgrid_path (str): Path to the TextGrid file.
        Returns:
            list: A list of tuples mapping syllables to their intervals [(syllable, start, end)].
        """
        consonants = set("ptkbdgmnlrsfvzʃʒɡʁjwŋtrɥgʀyc")
        vowels = set("aeɛioɔuøœəɑ̃ɛ̃ɔ̃œ̃ɑ̃ɔ̃ɑ̃ɔ̃")

        tg = tgio.openTextgrid(textgrid_path, includeEmptyIntervals=False)
        phone_tier = tg.getTier("phones")

        syllables = []
        i = 0

        while i < len(phone_tier.entries):
            start, end, phone = phone_tier.entries[i]
            if phone in vowels:
                syllables.append((phone, start, end))
                i += 1
            elif phone in consonants:
                if i + 1 < len(phone_tier.entries):
                    next_start, next_end, next_phone = phone_tier.entries[i + 1]
                    if next_phone in vowels and abs(end - next_start) < 0.01:  # Allow small time gap
                        syllable = phone + next_phone
                        syllables.append((syllable, start, next_end))
                        i += 2  # Skip the next phone since it's part of the syllable
                    else:
                        syllables.append((phone, start, end))
                        i += 1
                else:
                    syllables.append((phone, start, end))
                    i += 1
            else:
                print(f"Skipping phone: {phone}")
                i += 1

        return [(syl, start, end) for syl, start, end in syllables]
    def _add_audio(self, video_path, audio_path, output_path):
        """
        Add the original audio to the rendered video with robust error handling.
        """
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        try:
            with VideoFileClip(video_path) as video_clip, \
                AudioFileClip(audio_path) as audio_clip:

                # Handle audio/video duration mismatch
                if abs(video_clip.duration - audio_clip.duration) > 0.1:
                    audio_clip = audio_clip.subclip(0, video_clip.duration)

                final_clip = video_clip.set_audio(audio_clip)
                
                # Use more compatible H.264 codec with explicit parameters
                final_clip.write_videofile(
                    output_path,
                    codec="libx264",  # More widely supported than libx265
                    audio_codec="aac",
                    preset="fast",    # Faster encoding
                    ffmpeg_params=[
                        "-pix_fmt", "yuv420p",  # Ensure player compatibility
                        "-movflags", "+faststart"  # For web streaming
                    ],
                    threads=4,        # Limit threads to avoid resource issues
                    logger=None       # Disable verbose FFmpeg output
                )
                
            # Verify successful output
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file not created: {output_path}")
                
            return output_path

        except Exception as e:
            # Clean up partially written files
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(f"Failed to add audio: {str(e)}") from e
    
    def _render_video(self):
        """Render video with hand cues"""
        input_video = cv2.VideoCapture(self.config["video_path"])
        output_path = os.path.join(self.config["output_dir"], "rendered_video.mp4")
        
        frame_info = self._get_video_properties(input_video)
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            frame_info["fps"],
            (frame_info["width"], frame_info["height"])
        )

        for frame_idx in range(int(frame_info["frame_count"])):
            success, frame = input_video.read()
            if not success:
                break

            self.current_video_frame = frame
            current_time = frame_idx / frame_info["fps"]
            self._process_frame(current_time)
            video_writer.write(frame)

        input_video.release()
        video_writer.release()
        return output_path

    def _process_frame(self, current_time):
        """Add hand cues to a single frame"""
        rgb_frame = cv2.cvtColor(self.current_video_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            current_syllable = self._get_current_syllable(current_time)
            
            if current_syllable:
                self._render_hand_cue(face_landmarks, current_syllable)

    def _get_current_syllable(self, current_time):
        """Binary search for current syllable"""
        pos = bisect_left(self.syllable_times, current_time)
        if pos == 0:
            return None if current_time < self.syllable_map[0][1] else self.syllable_map[0][0]
        if pos == len(self.syllable_times):
            return None
        return self.syllable_map[pos-1][0]

    def _render_hand_cue(self, face_landmarks, syllable):
        """
        Render hand cue for a syllable using real hand images.
        Args:
            face_landmarks: MediaPipe face landmarks.
            syllable (str): The current syllable being processed.
        """
        # Map the syllable to its corresponding hand shape and position
        hand_shape, hand_pos = map_syllable_to_cue(syllable)
        
        # Load the preprocessed hand image
        hand_image = self._load_hand_image(hand_shape)
        
        # Calculate the scale factor based on the face size
        scale_factor = calculate_face_scale(face_landmarks, {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})
        
        # Determine the target position for the reference landmark
        target_x, target_y = self._get_target_position(face_landmarks, hand_pos)
        
        # Overlay the hand image on the current video frame
        self.current_video_frame = self._overlay_hand_image(
            self.current_video_frame,
            hand_image,
            target_x,
            target_y,
            scale_factor,
            hand_shape
        )

    def _load_hand_data(self, hand_shape: int) -> dict:
        """Load hand landmark data and reference face bbox from JSON"""
        hand_file = os.path.join(self.config['handshapes_dir'], f"handshape_{hand_shape}.json")
        
        if not os.path.exists(hand_file):
            raise FileNotFoundError(f"Hand shape {hand_shape} data missing: {hand_file}")
        
        with open(hand_file, 'r') as f:
            data = json.load(f)
            
        return {
            "landmarks": np.array(data["hand_landmarks"], dtype=np.float32),
            "face_bbox": data["face_bbox"]
        }
    
    def _get_reference_finger_position(self, hand_shape: int, reference_finger_idx: int):
        """
        Get the position of the reference finger in the original hand image.
        Args:
            hand_shape (int): Hand shape number (1 to 8).
            reference_finger_idx (int): Index of the reference finger.
        Returns:
            tuple: (x, y) coordinates of the reference finger in the original image.
        """
        # Load the hand landmarks for the specified hand shape
        hand_data = self._load_hand_data(hand_shape)
        hand_landmarks = hand_data["landmarks"]
        
        # Get the reference finger's position
        reference_finger_pos = hand_landmarks[reference_finger_idx][:2]  # Extract x, y coordinates
        
        return reference_finger_pos

    
    def _get_target_position(self, face_landmarks, hand_pos: int):
        """
        Get the target position for the reference landmark on the face.
        Args:
            face_landmarks: MediaPipe face landmarks.
            hand_pos (int): Target position index or special case (-1, -2).
        Returns:
            tuple: (target_x, target_y) in pixel coordinates.
        """
        frame_height, frame_width = self.current_video_frame.shape[:2]
        if hand_pos == -1:
            target_x = face_landmarks.landmark[57].x * frame_width - 0.06 * frame_width
            target_y = face_landmarks.landmark[57].y * frame_height
        elif hand_pos == -2:
            target_x = face_landmarks.landmark[118].x * frame_width
            target_y = face_landmarks.landmark[118].y * frame_height + 0.04 * frame_height
        else:
            target_x = face_landmarks.landmark[hand_pos].x * frame_width
            target_y = face_landmarks.landmark[hand_pos].y * frame_height
        return int(target_x), int(target_y)

    def _load_hand_image(self, hand_shape: int):
        """
        Load the preprocessed hand image for the specified hand shape.
        Args:
            hand_shape (int): Hand shape number (1 to 8).
        Returns:
            np.ndarray: Loaded hand image with transparency (RGBA).
        """
        hand_image_path = os.path.join(
            "/scratch2/bsow/Documents/ACSR/data/handshapes/rotated_images/",
            f"rotated_handshape_{hand_shape}.png"
        )
        if not os.path.exists(hand_image_path):
            raise FileNotFoundError(f"Hand image {hand_shape} not found: {hand_image_path}")
        return cv2.imread(hand_image_path, cv2.IMREAD_UNCHANGED)

    import pandas as pd

    def _overlay_hand_image(self, frame, hand_image, target_x, target_y, scale_factor, hand_shape):
        """
        Overlay the hand image on the current frame at the specified position and scale.
        Args:
            frame: Current video frame.
            hand_image: Preprocessed hand image with transparency.
            target_x, target_y: Target position for the reference finger.
            scale_factor: Scale factor for the hand image.
            hand_shape (int): The hand shape number (1 to 8).
        Returns:
            np.ndarray: Updated video frame with the hand image overlaid.
        """
        # Resize the hand image based on the scale factor
        h, w = hand_image.shape[:2]
        scaled_width = int(w * scale_factor * self.config["hand_scale_factor"])
        scaled_height = int(h * scale_factor * self.config["hand_scale_factor"])
        resized_hand = cv2.resize(hand_image, (scaled_width, scaled_height))
        
        # Load the reference finger's position from the CSV file
        csv_path = "/scratch2/bsow/Documents/ACSR/data/handshapes/yellow_pixels.csv"  # Update this path to your CSV file
        ref_finger_data = pd.read_csv(csv_path)
        
        # Filter the row corresponding to the current hand shape
        hand_row = ref_finger_data[ref_finger_data["image_name"] == f"handshape_{hand_shape}.png"]
        if hand_row.empty:
            raise ValueError(f"No reference finger data found for hand shape {hand_shape}")
        
        # Extract the reference finger's pixel coordinates
        ref_finger_x = hand_row["yellow_pixel_x"].values[0]  # Reference finger's x-coordinate
        ref_finger_y = hand_row["yellow_pixel_y"].values[0]  # Reference finger's y-coordinate
        
        # Scale the reference finger's position based on the scale factor
        ref_finger_x_scaled = ref_finger_x * scale_factor * self.config["hand_scale_factor"]
        ref_finger_y_scaled = ref_finger_y * scale_factor * self.config["hand_scale_factor"]
        
        # Calculate the top-left corner of the hand image
        x_offset = int(target_x - ref_finger_x_scaled)
        y_offset = int(target_y - ref_finger_y_scaled)
        
        # Ensure the hand image stays within the frame boundaries
        if x_offset < 0:
            x_offset = 0
        if y_offset < 0:
            y_offset = 0
        if x_offset + scaled_width > frame.shape[1]:
            x_offset = frame.shape[1] - scaled_width
        if y_offset + scaled_height > frame.shape[0]:
            y_offset = frame.shape[0] - scaled_height
        
        # Extract the alpha channel for transparency
        if resized_hand.shape[2] == 4:  # Check if the image has an alpha channel
            alpha_hand = resized_hand[:, :, 3] / 255.0
            alpha_frame = 1.0 - alpha_hand
            for c in range(0, 3):  # Apply alpha blending for RGB channels
                frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width, c] = (
                    alpha_hand * resized_hand[:, :, c] +
                    alpha_frame * frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width, c]
                )
        else:
            # If no alpha channel, simply overlay the hand image
            frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized_hand
        
        return frame

    def _choose_reference_finger(self, consonant_index):
        """
        Choose the reference finger (the one with the maximum distance between the base of the palm and the tip).
        Args:
            consonant_index (int): Index of the consonant.
        Returns:
            int: Index of the reference landmark.
        """
        if consonant_index == 1 or consonant_index == 6:
            return 8  # Tip of the index finger
        else:
            return 12  # Tip of the middle finger
            
    def _adjust_hand_position(self, hand_landmarks: np.ndarray, 
                          face_landmarks,
                          target_pos: int, scale_factor: float) -> np.ndarray:
        """Adjust hand position relative to face landmarks"""
        # Get target face landmark coordinates
        frame_height, frame_width = self.current_video_frame.shape[:2]
        
        if target_pos == -1:
            target_x = face_landmarks.landmark[57].x - 0.06
            target_y = face_landmarks.landmark[57].y
        elif target_pos == -2:
            target_x = face_landmarks.landmark[118].x
            target_y = face_landmarks.landmark[118].y + 0.04
        else:
            target_x = face_landmarks.landmark[target_pos].x
            target_y = face_landmarks.landmark[target_pos].y
        
        # Choose the reference finger
        reference_finger_idx = self._choose_reference_finger(target_pos)
        reference_landmark = hand_landmarks[reference_finger_idx]
        
        # Calculate the offset to move the reference finger to the target position
        offset_x = target_x - reference_landmark[0] * scale_factor * self.config["hand_scale_factor"]
        offset_y = target_y - reference_landmark[1] * scale_factor * self.config["hand_scale_factor"]
        
        # Adjust all hand landmarks by the offset
        adjusted = []
        for lm in hand_landmarks:
            scaled_x = lm[0] * scale_factor * self.config["hand_scale_factor"] * frame_width + offset_x * frame_width
            scaled_y = lm[1] * scale_factor * self.config["hand_scale_factor"] * frame_height + offset_y * frame_height
            adjusted.append([
                scaled_x,
                scaled_y,
                lm[2]  # Maintain depth coordinate
            ])
        return np.array(adjusted, dtype=np.float32)

    def _draw_hand(self, landmarks: np.ndarray):
        """
        Draw hand landmarks on the current frame with connections.
        Args:
            landmarks (np.ndarray): Adjusted hand landmarks (already in pixel space).
        """
        frame_height, frame_width = self.current_video_frame.shape[:2]
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            x1, y1 = int(landmarks[start_idx][0]), int(landmarks[start_idx][1])
            x2, y2 = int(landmarks[end_idx][0]), int(landmarks[end_idx][1])
            
            # Ensure the points are within the frame boundaries
            if 0 <= x1 < frame_width and 0 <= y1 < frame_height and \
            0 <= x2 < frame_width and 0 <= y2 < frame_height:
                cv2.line(self.current_video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks
        for lm in landmarks:
            x, y = int(lm[0]), int(lm[1])
            
            # Ensure the point is within the frame boundaries
            if 0 <= x < frame_width and 0 <= y < frame_height:
                cv2.circle(self.current_video_frame, (x, y), 5, (0, 0, 255), -1)


    def _get_video_properties(self, cap: cv2.VideoCapture) -> dict:
        """Get essential video properties"""
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
if __name__ == "__main__":
    processor = CuedSpeechProcessor(CONFIG)
    processor.process_video()