import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Padding function to ensure videos are the same length
def pad_sequence(sequence, max_length, padding_value=0):
    padding_size = max_length - len(sequence)
    if padding_size > 0:
        padding = np.zeros((padding_size, *sequence.shape[1:]), dtype=sequence.dtype)
        sequence = np.concatenate([sequence, padding], axis=0)
    return torch.tensor(sequence).clone().detach()

# Dataset class for loading video data
class VideoDataset(Dataset):
    def __init__(self, videos, labels, max_length=None):
        self.videos = videos
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        label = self.labels[idx]
        if self.max_length:
            video = pad_sequence(video, self.max_length)  # Apply padding
        return video.clone().detach(), torch.tensor(label).clone().detach()

# Custom collate function for batching videos and padding them
def collate_fn(batch):
    videos, labels = zip(*batch)
    max_length = max(len(video) for video in videos)  # Find max length of videos in this batch
    padded_videos = [pad_sequence(video, max_length) for video in videos]  # Apply padding
    return torch.stack(padded_videos), torch.tensor(labels)

# Example videos and labels (replace with your actual dataset)
videos = [np.random.rand(10, 3, 16, 16).astype(np.float32) for _ in range(100)]  # Dummy video data
labels = [np.random.randint(0, 10) for _ in range(100)]  # Dummy labels

# Define the maximum length for padding (e.g., 20 frames)
max_length = 20
batch_size = 16

# Initialize the train_dataset and train_loader
train_dataset = VideoDataset(videos, labels, max_length=max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Define a simple neural network for demonstration
class VideoModel(nn.Module):
    def __init__(self, num_classes):
        super(VideoModel, self).__init__()
        self.conv = nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 16 * 16 * max_length, num_classes)  # Adjust dimensions

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# Initialize model, loss function, and optimizer
num_classes = 10  # Replace with the actual number of classes in your dataset
model = VideoModel(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 5
save_path = "C:\\Users\\pc\\Downloads\\modeltrain\\model\\model.pth"  # Define the path to save the model

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for videos, labels in train_loader:
        optimizer.zero_grad()
        # Ensure videos have the correct dimensions (batch_size, channels, depth, height, width)
        videos = videos.permute(0, 2, 1, 3, 4)  # Rearrange dimensions: (batch_size, time, channels, height, width) -> (batch_size, channels, time, height, width)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")