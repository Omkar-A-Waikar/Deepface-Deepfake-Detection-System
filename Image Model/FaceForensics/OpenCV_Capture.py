import cv2
import os
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor

# Define input and output paths
original_videos_path = Path("F:/FF_Dataset/original_sequences/actors/c23/videos")
manipulated_videos_path = Path("F:/FF_Dataset/manipulated_sequences/DeepFakeDetection/c23/videos")
output_dataset_path = Path("F:/FF_Dataset/processed_frames")
interval_in_seconds = 4  # Interval in seconds for extracting frames
train_ratio = 0.8  # Training set ratio

# Create output directories
(output_dataset_path / "train" / "REAL").mkdir(parents=True, exist_ok=True)
(output_dataset_path / "train" / "FAKE").mkdir(parents=True, exist_ok=True)
(output_dataset_path / "test" / "REAL").mkdir(parents=True, exist_ok=True)
(output_dataset_path / "test" / "FAKE").mkdir(parents=True, exist_ok=True)

def extract_frames(video_path, output_folder, label, split, interval_in_seconds=4):
    """
    Extract frames from a video at specified intervals and save them to the specified folder.
    :param video_path: Path to the video file
    :param output_folder: Path to the output folder
    :param label: Label ('REAL' or 'FAKE')
    :param split: 'train' or 'test', indicating which set to save frames into
    :param interval_in_seconds: Number of seconds between each extracted frame
    """
    video_capture = cv2.VideoCapture(str(video_path))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_in_seconds)  # Calculate the interval in frames
    frame_count = 0
    success, frame = video_capture.read()

    split_output_folder = output_folder / split / label

    while success:
        if frame_count % interval == 0:
            frame_filename = f"{video_path.stem}_frame_{frame_count:04d}.jpg"
            frame_output_path = split_output_folder / frame_filename
            cv2.imwrite(str(frame_output_path), frame)
        success, frame = video_capture.read()
        frame_count += 1
    video_capture.release()

def process_video(video_file, label, split, interval_in_seconds=2):
    """
    Wrapper function to process a single video, used for multithreading.
    """
    extract_frames(video_file, output_dataset_path, label, split, interval_in_seconds)

# Get list of all video files
original_videos = list(original_videos_path.glob("*.mp4"))
manipulated_videos = list(manipulated_videos_path.glob("*.mp4"))

# Shuffle and split videos into training and testing sets
random.shuffle(original_videos)
random.shuffle(manipulated_videos)

train_original_videos = original_videos[:int(len(original_videos) * train_ratio)]
test_original_videos = original_videos[int(len(original_videos) * train_ratio):]

train_manipulated_videos = manipulated_videos[:int(len(manipulated_videos) * train_ratio)]
test_manipulated_videos = manipulated_videos[int(len(manipulated_videos) * train_ratio):]

# Define the number of threads to use
num_threads = 12  # You can adjust this based on your system's capabilities

# Process videos with multithreading
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit tasks for training set (REAL videos)
    for video_file in train_original_videos:
        executor.submit(process_video, video_file, "REAL", "train", interval_in_seconds)

    # Submit tasks for testing set (REAL videos)
    for video_file in test_original_videos:
        executor.submit(process_video, video_file, "REAL", "test", interval_in_seconds)

    # Submit tasks for training set (FAKE videos)
    for video_file in train_manipulated_videos:
        executor.submit(process_video, video_file, "FAKE", "train", interval_in_seconds)

    # Submit tasks for testing set (FAKE videos)
    for video_file in test_manipulated_videos:
        executor.submit(process_video, video_file, "FAKE", "test", interval_in_seconds)

print("Frame extraction complete!")
