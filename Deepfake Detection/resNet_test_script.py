import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import timm
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_frames=100):
        self.data_dir = data_dir
        self.transform = transform
        self.video_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.mp4')]
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        frames = self.read_frames(video_path)
        frames = self.pad_or_trim_frames(frames, self.max_frames)

        if self.transform:
            transformed_frames = [self.transform(frame) for frame in frames]
            transformed_frames = torch.stack(transformed_frames)
        else:
            transformed_frames = torch.stack(frames)

        return transformed_frames

    def read_frames(self, video_path):
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            cap.release()
        except Exception as e:
            print(f"Error reading video file: {video_path}")
            print(e)
        return frames

    def pad_or_trim_frames(self, frames, max_frames):
        if len(frames) >= max_frames:
            frames = frames[:max_frames]
        else:
            last_frame = frames[-1]
            frames.extend([last_frame] * (max_frames - len(frames)))
        return frames

def load_model(model_name='efficientnet_b0', num_classes=2, model_path='deepfake_detection_model_epoch6.pth'):
    model = timm.create_model(model_name, pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def test_model(model, data_loader):
    model.eval()
    device = next(model.parameters()).device  # Get the device of the model
    predictions = []
    
    with torch.no_grad():
        for inputs in data_loader:
            # Flatten the inputs to combine batch and frame dimensions
            batch_size, num_frames, channels, height, width = inputs.shape
            inputs = inputs.view(batch_size * num_frames, channels, height, width)
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Reshape outputs back to batch and frame dimensions
            outputs = outputs.view(batch_size, num_frames, -1)
            
            # Take the mean across frames to get a single prediction per video
            outputs = torch.mean(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return predictions


if __name__ == "__main__":
    # Define paths and parameters relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(script_dir, 'data', 'test2')
    model_path = os.path.join(script_dir, 'deepfake_detection_model_epoch6.pth')

    # Define batch size and other DataLoader parameters
    batch_size = 4
    max_frames = 50

    # Define transformations for video frames
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Reduce image size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create custom dataset and data loader for test data
    test_dataset = VideoDataset(test_data_dir, transform=transform, max_frames=max_frames)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path=model_path)
    model = model.to(device)

    # Test the model on unseen videos
    predictions = test_model(model, test_loader)

    # Count the number of predicted deepfake and real videos
    num_deepfakes = sum(predictions)
    num_real = len(predictions) - num_deepfakes

    # Compute percentages
    total_videos = len(predictions)
    percentage_deepfakes = (num_deepfakes / total_videos) * 100
    percentage_real = (num_real / total_videos) * 100

    # Output predictions and percentages
    print("Predictions:")
    print(predictions)
    print(f"Percentage of deepfake videos: {percentage_deepfakes:.2f}%")
    print(f"Percentage of real videos: {percentage_real:.2f}%")
