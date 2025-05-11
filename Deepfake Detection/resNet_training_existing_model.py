import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import timm
from resNet_deepfake_detection import VideoDataset  # Import your custom dataset class
import multiprocessing

def main():
    # Define paths to training and validation data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', 'train')
    val_data_dir = os.path.join(script_dir, 'data', 'validation')

    # Define batch size and other DataLoader parameters
    batch_size = 4
    max_frames = 50

    # Define transformations for video frames
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize image
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create custom datasets and data loaders
    train_dataset = VideoDataset(data_dir, transform=transform, max_frames=max_frames)
    val_dataset = VideoDataset(val_data_dir, transform=transform, max_frames=max_frames)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Set num_workers to 0
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # Set num_workers to 0

    # Load the pre-trained model and move it to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(script_dir, 'deepfake_detection_model_epoch5.pth')

    # Create the same model architecture as during training
    model = timm.create_model('efficientnet_b0', pretrained=False)  # Change model_name as needed
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)  # Adjust num_classes based on your task
    model = model.to(device)

    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5  # Adjust as needed
    best_val_loss = float('inf')
    patience = 3
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

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

        # Validate the model after each epoch
        with torch.no_grad():
            val_loss = 0.0
            model.eval()
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                batch_size, num_frames, channels, height, width = val_inputs.shape
                val_inputs = val_inputs.view(batch_size * num_frames, channels, height, width)
                val_outputs = model(val_inputs)
                val_outputs = val_outputs.view(batch_size, num_frames, -1)
                val_outputs = torch.mean(val_outputs, dim=1)
                val_loss += criterion(val_outputs, val_labels).item()

            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f'Early stopping triggered. Validation loss did not improve for {patience} epochs.')
                    break

        # Save the model checkpoint after each epoch
        torch.save(model.state_dict(), f'deepfake_detection_model_epoch{epoch + 6}.pth')
        print(f'Model saved for epoch {epoch + 1}\n')

    print('Finished training.')

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Properly initialize multiprocessing on Windows
    main()
