import cv2
import os

class VideoDataGenerator:
    def __init__(self, video_path):
        self.video_path = video_path

    def get_video_info(self):
        cap = cv2.VideoCapture(self.video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        channels = 3  # Assuming standard RGB video
        cap.release()
        return num_frames, (height, width, channels)

# Path to your video file
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'data', 'train2', 'WIN_20240511_17_01_32_Pro.mp4')
# Create an instance of VideoDataGenerator
generator = VideoDataGenerator(video_path)

# Get video information: number of frames and image shape dimensions
num_frames, image_shape = generator.get_video_info()

if num_frames is not None and image_shape is not None:
    print(f"Number of Frames: {num_frames}")
    print(f"Image Shape: {image_shape}")
else:
    print("Error: Unable to retrieve video information.")
