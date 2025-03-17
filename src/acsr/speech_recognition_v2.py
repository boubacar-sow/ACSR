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

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

class CuedSpeechProcessor:
    def __init__(self, config):
        self.config = config
        self.syllable_map = []
        self.current_video_frame = None
        self._validate_paths()
        self.current_hand_pos = None
        self.target_hand_pos = None
        self.active_transition = None
        self.last_active_syllable = None

    def _validate_paths(self):
        """Ensure required directories and files exist."""
        if not os.path.exists(self.config["video_path"]):
            raise FileNotFoundError(f"Video file not found: {self.config['video_path']}")
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def process_video(self):
        """Main processing pipeline."""
        try:
            audio_path = self._extract_audio()
            transcription = self._transcribe_audio(audio_path)
            self._align_and_build_syllables(audio_path, transcription)
            rendered_video = self._render_video()
            final_output = self._add_audio(rendered_video, audio_path)
            logging.info(f"Processing complete. Final output: {final_output}")
            return final_output
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            raise

    def _extract_audio(self):
        """Extract audio from video file."""
        audio_path = os.path.join(self.config["output_dir"], "audio.wav")
        with VideoFileClip(self.config["video_path"]) as video:
            video.audio.write_audiofile(audio_path, codec="pcm_s16le")
        logging.info(f"Audio extracted to {audio_path}")
        return audio_path

    def _transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("medium", device=device)
        result = model.transcribe(audio_path, language=self.config["language"])
        logging.info("Audio transcription completed")
        return result["text"]

    def _align_and_build_syllables(self, audio_path, text):
        """Align text and build syllable timeline"""
        textgrid_path = "/scratch2/bsow/Documents/ACSR/data/transcriptions/cuedspeech2.TextGrid"
        self.syllable_map = self._parse_textgrid(textgrid_path)
        
        # Sort by start time using 'a1' key instead of tuple index
        self.syllable_map.sort(key=lambda x: x['a1'])
        
        # Create syllable times list using dictionary keys
        self.syllable_times = [item['a1'] for item in self.syllable_map]
        print(self.syllable_map)

    def _run_mfa_alignment(self, audio_path, text):
        """Run Montreal Forced Aligner"""
        # Create a temporary directory for MFA input
        mfa_input_dir = os.path.join(self.config['output_dir'], "mfa_input")
        os.makedirs(mfa_input_dir, exist_ok=True)
        audio_filename = os.path.basename(audio_path)
        mfa_audio_path = os.path.join(mfa_input_dir, audio_filename)
        os.system(f"cp {audio_path} {mfa_audio_path}")
        text_filename = os.path.splitext(audio_filename)[0] + ".lab"
        text_path = os.path.join(mfa_input_dir, text_filename)
        with open(text_path, "w") as f:
            f.write(text)
        # Build MFA command
        cmd = ["mfa", "align", mfa_input_dir, f"{self.config['language']}_mfa",
            f"{self.config['language']}_mfa", self.config["output_dir"], "--clean"
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
        consonants = "ptkbdgmnlrsfvzʃʒɡʁjwŋtrɥgʀcɲ"
        vowels = "aeɛioɔuøœyəɑ̃ɛ̃ɔ̃œ̃ɑ̃ɔ̃ɑ̃ɔ̃"
        
        tg = tgio.openTextgrid(textgrid_path, includeEmptyIntervals=False)
        phone_tier = tg.getTier("phones")
        syllables = []
        i = 0
        while i < len(phone_tier.entries):
            start, end, phone = phone_tier.entries[i]
            phone = list(phone)
            if len(phone) == 2:
                if phone[0] in vowels and phone[1] == "̃":
                    syllables.append((phone[0] + phone[1], start, end))
                    i += 1
            else:
                if phone[0] in vowels:
                    syllables.append((phone[0], start, end))
                    i += 1
                elif phone[0] in consonants:
                    if i + 1 < len(phone_tier.entries):
                        next_start, next_end, next_phone = phone_tier.entries[i + 1]
                        next_phone = list(next_phone)
                        if len(next_phone) == 2:
                            if next_phone[0] in vowels and abs(end - next_start) < 0.01 and next_phone[1] == "̃":
                                syllables.append((phone[0] + next_phone[0] + next_phone[1], start, next_end))
                                i += 2
                        else:
                            if next_phone[0] in vowels and abs(end - next_start) < 0.01:
                                syllables.append((phone[0] + next_phone[0], start, next_end))
                                i += 2
                            else:
                                syllables.append((phone[0], start, end))
                                i += 1
                    else:
                        syllables.append((phone[0], start, end))
                        i += 1
                else:
                    print(f"Skipping phone: {phone[0]}, len phone {len(phone)}")
                    i += 1
        enhanced_syllables = []
        prev_syllable_end = 0
        for i, (syllable, start, end) in enumerate(syllables):
            print(syllable, start, end)
            # Determine syllable type
            if len(syllable) == 1:
                syl_type = 'C' if syllable in consonants else 'V'
            else:
                syl_type = 'CV'
            
            # Calculate A1A3 duration in seconds
            a1a3_duration = end - start
            
            # Determine context
            from_neutral = (i == 0 or (start - prev_syllable_end) > 0.5)  # If pause >500ms
            to_neutral = False  # Implement similar logic for end of utterance
            
            # Calculate M1 and M2 based on WP3 algorithm
            if from_neutral:
                m1 = start - (a1a3_duration * 1.60)
                m2 = start - (a1a3_duration * 0.10)
            elif to_neutral:
                m1 = start - 0.03
                m2 = m1 + 0.37
            else:
                if syl_type == 'C':
                    m1 = start - (a1a3_duration * 1.60)
                    m2 = start - (a1a3_duration * 0.30)
                elif syl_type == 'V':
                    m1 = start - (a1a3_duration * 2.40)
                    m2 = start - (a1a3_duration * 0.60)
                else:  # CV
                    m1 = start - (a1a3_duration * 0.80)
                    m2 = start + (a1a3_duration * 0.11)
            
            enhanced_syllables.append({
                'syllable': syllable,
                'a1': start,
                'a3': end,
                'm1': m1,
                'm2': m2,
                'type': syl_type
            })
            prev_syllable_end = end

        # In your _parse_textgrid method, after calculating m1 and m2:
        MIN_DISPLAY_DURATION = 0.4  # 250ms minimum display time
        for i in range(len(enhanced_syllables)):
            syl = enhanced_syllables[i]
            current_duration = syl['m2'] - syl['m1']
            
            # If transition is too fast, extend it
            if current_duration < MIN_DISPLAY_DURATION:
                # For consonants, prioritize extending
                extension_needed = MIN_DISPLAY_DURATION - current_duration
                syl['m2'] = syl['m1'] + MIN_DISPLAY_DURATION

        # After building enhanced_syllables, enforce non-overlapping transition windows.
        for i in range(1, len(enhanced_syllables)):
            prev = enhanced_syllables[i - 1]
            curr = enhanced_syllables[i]
            # Ensure that the new syllable's m1 is not before the previous syllable's m2.
            if curr['m1'] < prev['m2']:
                # Adjust m1 upward to the previous m2.
                curr['m1'] = prev['m2']
                # Optionally, adjust m2 to maintain the same transition window duration ratio:
                duration = curr['m2'] - curr['m1']
                # Here you might choose a strategy (e.g., keep m2 unchanged or set m2 = m1 + original_duration)
                # For now, we simply leave m2 as is.
        
        
        return enhanced_syllables

    def _add_audio(self, video_path, audio_path):
        """
        Add the original audio to the rendered video with robust error handling.
        """
        output_path = os.path.join(self.config["output_dir"], f"final_{os.path.basename(video_path)}")
        try:
            with VideoFileClip(video_path) as video_clip, AudioFileClip(audio_path) as audio_clip:
                if abs(video_clip.duration - audio_clip.duration) > 0.1:
                    audio_clip = audio_clip.subclip(0, video_clip.duration)
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    preset="fast",
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                    threads=4,
                    logger=None
                )
                if not os.path.exists(output_path):
                    raise RuntimeError(f"Output file not created: {output_path}")
                return output_path
        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(f"Failed to add audio: {str(e)}") from e

    def _render_video(self):
        """Render video with hand cues."""
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
        rgb_frame = cv2.cvtColor(self.current_video_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Find active transition for the current time
            self.active_transition = None
            for syl in self.syllable_map:
                if syl['m1'] <= current_time <= syl['m2']:
                    self.active_transition = syl
                    break
            
            if self.active_transition:
                progress = (current_time - self.active_transition['m1']) / (self.active_transition['m2'] - self.active_transition['m1'])
                self._render_hand_transition(face_landmarks, progress)
            else:
                # If no new gesture is active, persist the last hand position 
                # as long as the last syllable is not the final syllable of the sentence.
                if self.current_hand_pos is not None and \
                self.last_active_syllable is not None and \
                self.last_active_syllable != self.syllable_map[-1]:
                    hand_shape, hand_pos_code = map_syllable_to_cue(self.last_active_syllable['syllable'])
                    hand_image = self._load_hand_image(hand_shape)
                    scale_factor = calculate_face_scale(face_landmarks, {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})
                    self.current_video_frame = self._overlay_hand_image(
                        self.current_video_frame,
                        hand_image,
                        self.current_hand_pos[0],
                        self.current_hand_pos[1],
                        scale_factor,
                        hand_shape
                    )

    def _render_hand_transition(self, face_landmarks, progress):
            progress = max(0.0, min(1.0, progress))
            target_shape, hand_pos_code = map_syllable_to_cue(self.active_transition['syllable'])
            final_target = self._get_target_position(face_landmarks, hand_pos_code)
            
            if self.current_hand_pos is None:
                self.current_hand_pos = final_target
                self.last_active_syllable = self.active_transition  # <<-- Store last syllable
                return
            
            
            new_x = self.current_hand_pos[0] + (final_target[0] - self.current_hand_pos[0]) * progress
            new_y = self.current_hand_pos[1] + (final_target[1] - self.current_hand_pos[1]) * progress
            intermediate_pos = (int(new_x), int(new_y))
            
            if progress < 0.95:
                self.current_hand_pos = intermediate_pos
            else:
                self.current_hand_pos = final_target
            
            hand_image = self._load_hand_image(target_shape)
            scale_factor = calculate_face_scale(face_landmarks, {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})
            self.current_video_frame = self._overlay_hand_image(
                self.current_video_frame,
                hand_image,
                intermediate_pos[0],
                intermediate_pos[1],
                scale_factor,
                target_shape
            )
            
            # Store the syllable that is currently active
            self.last_active_syllable = self.active_transition

    def _get_current_syllable(self, current_time):
        """Binary search for current syllable."""
        pos = bisect_left(self.syllable_times, current_time)
        if pos == 0:
            return None if current_time < self.syllable_map[0]['a1'] else self.syllable_map[0]['syllable']
        if pos == len(self.syllable_times):
            return None
        return self.syllable_map[pos - 1]['syllable']

    def _render_hand_cue(self, face_landmarks, syllable):
        """
        Render hand cue for a syllable using real hand images.
        Args:
            face_landmarks: MediaPipe face landmarks.
            syllable (str): The current syllable being processed.
        """
        hand_shape, hand_pos = map_syllable_to_cue(syllable)
        hand_image = self._load_hand_image(hand_shape)
        scale_factor = calculate_face_scale(face_landmarks, {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})
        target_x, target_y = self._get_target_position(face_landmarks, hand_pos)
        self.current_video_frame = self._overlay_hand_image(
            self.current_video_frame,
            hand_image,
            target_x,
            target_y,
            scale_factor,
            hand_shape
        )

    def _load_hand_image(self, hand_shape):
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
        h, w = hand_image.shape[:2]
        scaled_width = int(w * scale_factor * self.config["hand_scale_factor"])
        scaled_height = int(h * scale_factor * self.config["hand_scale_factor"])
        resized_hand = cv2.resize(hand_image, (scaled_width, scaled_height))

        csv_path = "/scratch2/bsow/Documents/ACSR/data/handshapes/yellow_pixels.csv"
        ref_finger_data = pd.read_csv(csv_path)
        hand_row = ref_finger_data[ref_finger_data["image_name"] == f"handshape_{hand_shape}.png"]
        if hand_row.empty:
            raise ValueError(f"No reference finger data found for hand shape {hand_shape}")

        ref_finger_x = hand_row["yellow_pixel_x"].values[0]
        ref_finger_y = hand_row["yellow_pixel_y"].values[0]
        ref_finger_x_scaled = ref_finger_x * scale_factor * self.config["hand_scale_factor"]
        ref_finger_y_scaled = ref_finger_y * scale_factor * self.config["hand_scale_factor"]

        x_offset = int(target_x - ref_finger_x_scaled)
        y_offset = int(target_y - ref_finger_y_scaled)

        # Ensure the hand image stays within the frame boundaries
        x_offset = max(0, min(x_offset, frame.shape[1] - scaled_width))
        y_offset = max(0, min(y_offset, frame.shape[0] - scaled_height))

        if resized_hand.shape[2] == 4:  # Check if the image has an alpha channel
            alpha_hand = resized_hand[:, :, 3] / 255.0
            alpha_frame = 1.0 - alpha_hand
            for c in range(3):
                frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width, c] = (
                    alpha_hand * resized_hand[:, :, c] +
                    alpha_frame * frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width, c]
                )
        else:
            frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized_hand
        return frame

    def _get_target_position(self, face_landmarks, hand_pos):
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
            target_x = face_landmarks.landmark[152].x * frame_width
            target_y = face_landmarks.landmark[152].y * frame_height + 0.06 * frame_height
        else:
            target_x = face_landmarks.landmark[hand_pos].x * frame_width
            target_y = face_landmarks.landmark[hand_pos].y * frame_height
        return int(target_x), int(target_y)

    def _get_video_properties(self, cap):
        """Get essential video properties."""
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

if __name__ == "__main__":
    processor = CuedSpeechProcessor(CONFIG)
    processor.process_video()