import os
import subprocess
from glob import glob

# Input and output directories
input_dir = "/scratch2/bsow/Documents/ACSR/data/training_videos/CSF22_test/test_videos"
output_dir = "/scratch2/bsow/Documents/ACSR/data/training_videos/CSF22_test/mp4"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert .webm to .mp4 and change frame rate to 30 fps
def convert_video(video_path, output_path):
    command = [
        "ffmpeg",
        "-i", video_path,  # Input video file
        "-r", "30",        # Set frame rate to 30 fps
        "-c:v", "libx264", # Video codec
        "-c:a", "aac",     # Audio codec
        "-b:a", "192k",    # Audio bitrate
        "-vf", "fps=30",   # Ensure frame rate is 30 fps
        output_path,       # Output video file
    ]
    subprocess.run(command, check=True)
    print(f"Converted and saved to: {output_path}")

# Main function
def main():
    # Find all .webm files in the input directory and its subdirectories
    video_files = glob(os.path.join(input_dir, "**/*.webm"), recursive=True)
    
    # Process each video
    for video_path in video_files:
        # Get the relative path to maintain the folder structure
        relative_path = os.path.relpath(video_path, input_dir)
        output_path = os.path.join(output_dir, relative_path.replace(".webm", ".mp4"))
        
        # Create the output subdirectory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert the video
        convert_video(video_path, output_path)

if __name__ == "__main__":
    main()