import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from resNet_deepfake_detection import VideoDataset  # Import your custom dataset class
from torchvision.transforms import transforms
import os
import timm  # Import the timm library for pretrained models

# Define paths to validation data and saved models
script_dir = os.path.dirname(os.path.abspath(__file__))
val_data_dir = os.path.join(script_dir, 'data', 'test3')
model_paths = [
    os.path.join(script_dir, 'deepfake_detection_model_epoch7.pth')
]

# Define batch size and other DataLoader parameters
batch_size = 4
max_frames = 50

# Define transformations for video frames (make sure to use the same transforms as during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create validation dataset and data loader
val_dataset = VideoDataset(val_data_dir, transform=transform, max_frames=max_frames)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Function to evaluate model on validation dataset
def evaluate_model(model, criterion, device):
    model.eval()
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for val_inputs, labels in val_loader:
            val_inputs, labels = val_inputs.to(device), labels.to(device)
            batch_size, num_frames, channels, height, width = val_inputs.shape
            val_inputs = val_inputs.view(batch_size * num_frames, channels, height, width)
            outputs = model(val_inputs)
            outputs = outputs.view(batch_size, num_frames, -1)
            outputs = torch.mean(outputs, dim=1)
            val_loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds)
    recall = recall_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

if __name__ == "__main__":
    # Load each model and evaluate its performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    for model_path in model_paths:
        # Create the same model architecture as during training
        model = timm.create_model('efficientnet_b0', pretrained=False)  # Change model_name as needed
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)  # Adjust num_classes based on your task
        model = model.to(device)

        # Load the model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Evaluate the model
        evaluation_metrics = evaluate_model(model, criterion, device)
        print(f"Evaluation metrics for model {model_path}:")
        print(evaluation_metrics)
