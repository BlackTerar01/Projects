import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import timm
from PIL import Image
import time

# Custom dataset class for loading video frames
class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_frames=100):
        self.data_dir = data_dir
        self.transform = transform
        self.video_files = []
        self.labels = []
        self.max_frames = max_frames
        
        label_map = {
            'real_video_real_audio': 0,
            'fake_video_real_audio': 1,
            'real_video_fake_audio': 0,
            'fake_video_fake_audio': 1
        }
        
        for label in label_map:
            label_path = os.path.join(data_dir, label)
            if os.path.isdir(label_path):
                for video_file in os.listdir(label_path):
                    video_path = os.path.join(label_path, video_file)
                    # Skip files that are not valid video files
                    if not video_file.endswith('.ini'):
                        self.video_files.append(video_path)
                        self.labels.append(label_map[label])
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        frames = self.read_frames(video_path)
        frames = self.pad_or_trim_frames(frames, self.max_frames)
        
        if self.transform:
            transformed_frames = [self.transform(frame) for frame in frames]
            transformed_frames = torch.stack(transformed_frames)
        else:
            transformed_frames = torch.stack(frames)
        
        return transformed_frames, label
    
    def read_frames(self, video_path):
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video file: {video_path}")
                return frames
            
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
        if not frames:
            print(f"No frames were extracted from the video.")
            return frames
        
        if len(frames) >= max_frames:
            frames = frames[:max_frames]
        else:
            last_frame = frames[-1]
            frames.extend([last_frame] * (max_frames - len(frames)))
        
        return frames


def create_model(model_name='efficientnet_b0', pretrained=True, num_classes=2):
    model = timm.create_model(model_name, pretrained=pretrained)
    
    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model


def validate(model, val_loader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            # Reshape input to merge batch and frame dimensions
            batch_size, num_frames, channels, height, width = val_inputs.shape
            val_inputs = val_inputs.view(batch_size * num_frames, channels, height, width)
            val_outputs = model(val_inputs)
            val_outputs = val_outputs.view(batch_size, num_frames, -1)
            val_outputs = torch.mean(val_outputs, dim=1)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item()
    
    return val_running_loss / len(val_loader)


if __name__ == "__main__":

    # Define paths to training and validation data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', 'train')
    val_data_dir = os.path.join(script_dir, 'data', 'validation')

    # Define batch size and other DataLoader parameters
    batch_size = 4
    max_frames = 50

    # Define transformations for video frames
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Reduce image size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create custom datasets and data loaders
    train_dataset = VideoDataset(data_dir, transform=transform, max_frames=max_frames)
    val_dataset = VideoDataset(val_data_dir, transform=transform, max_frames=max_frames)

    # Use num_workers to enable data prefetching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load a pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    best_val_loss = float('inf')
    patience = 3
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Reshape inputs to merge batch and frame dimensions
            batch_size, num_frames, channels, height, width = inputs.shape
            inputs = inputs.view(batch_size * num_frames, channels, height, width)
            
            outputs = model(inputs)
            
            # Reshape outputs to match original batch and frame dimensions
            outputs = outputs.view(batch_size, num_frames, -1)
            outputs = torch.mean(outputs, dim=1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_time:.2f} seconds')
        
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Early stopping triggered. Validation loss did not improve for {patience} epochs.')
                break
        
        torch.save(model.state_dict(), f'deepfake_detection_model_epoch{epoch + 1}.pth')
        print(f'Model saved for epoch {epoch + 1}\n')
