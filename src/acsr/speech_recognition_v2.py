import json
import logging
import os
import subprocess
from bisect import bisect_left

import cv2
import mediapipe as mp
import numpy as np
import torch
import whisper
from moviepy.editor import AudioFileClip, VideoFileClip
from praatio import textgrid as tgio

from speech_recognition_v1 import (calculate_face_scale, 
                                   map_syllable_to_cue)
from variables import HAND_CONNECTIONS, consonant_to_handshape, vowel_positions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CONFIG = {
    "video_path": "/scratch2/bsow/Documents/ACSR/data/videos/cuedspeech2.mp4",
    "output_dir": "/scratch2/bsow/Documents/ACSR/data/transcriptions",
    "handshapes_dir": "/scratch2/bsow/Documents/ACSR/data/handshapes/coordinates",
    "language": "french",
    "reference_face_size": 0.3,  # Normalized reference face size
    "hand_scale_factor": 0.45,
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
        vowels = set("aeɛioɔuøœəɑ̃ɛ̃ɔ̃œ̃ɑ̃ɔ̃")

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
    def _add_audio(video_path, audio_path, output_path):
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
        """Render hand cue for a syllable"""
        hand_shape, hand_pos = map_syllable_to_cue(syllable)
        hand_data = self._load_hand_data(hand_shape)
        
        scale_factor = calculate_face_scale(face_landmarks, hand_data["face_bbox"])
        adjusted_landmarks = self._adjust_hand_position(
            hand_data["landmarks"], 
            face_landmarks, 
            hand_pos, 
            scale_factor
        )
        
        self._draw_hand(adjusted_landmarks)

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

    def _adjust_hand_position(self, hand_landmarks: np.ndarray, 
                            face_landmarks,
                            target_pos: int, scale_factor: float) -> np.ndarray:
        """Adjust hand position relative to face landmarks"""
        # Get target face landmark coordinates
        target_lm = face_landmarks.landmark[target_pos]
        frame_height, frame_width = self.current_video_frame.shape[:2]
        
        # Convert normalized coordinates to pixels
        target_x = int(target_lm.x * frame_width)
        target_y = int(target_lm.y * frame_height)
        
        # Scale and shift hand landmarks
        adjusted = []
        for lm in hand_landmarks:
            scaled_x = lm[0] * scale_factor * self.config["hand_scale_factor"]
            scaled_y = lm[1] * scale_factor * self.config["hand_scale_factor"]
            adjusted.append([
                scaled_x + target_x,
                scaled_y + target_y,
                lm[2]  # Maintain depth coordinate
            ])
            
        return np.array(adjusted, dtype=np.float32)

    def _draw_hand(self, landmarks: np.ndarray):
        """Draw hand landmarks on current frame with connections"""
        frame_height, frame_width = self.current_video_frame.shape[:2]
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            x1 = int(landmarks[start_idx][0] * frame_width)
            y1 = int(landmarks[start_idx][1] * frame_height)
            x2 = int(landmarks[end_idx][0] * frame_width)
            y2 = int(landmarks[end_idx][1] * frame_height)
            cv2.line(self.current_video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks
        for lm in landmarks:
            x = int(lm[0] * frame_width)
            y = int(lm[1] * frame_height)
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