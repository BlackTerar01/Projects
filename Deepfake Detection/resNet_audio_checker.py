import os
import torch
import torchvision.io as io
from moviepy.editor import VideoFileClip

# Path to the video file
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'data', 'train', 'real_video_fake_audio', '00001_fake.mp4')

# Load the video file using torchvision
video_frames, audio_metadata,x = io.read_video(video_path)

# Extract audio from the video using moviepy
clip = VideoFileClip(video_path)
audio_file_path = "extracted_audio.wav"
clip.audio.write_audiofile(audio_file_path)

# Load the extracted audio using torchaudio
import torchaudio
waveform, sample_rate = torchaudio.load(audio_file_path)

# Display audio information
print(f"Sample rate: {sample_rate}")
print(f"Number of channels: {waveform.shape[0]}")
print(f"Duration (seconds): {waveform.shape[1] / sample_rate}")

