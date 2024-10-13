import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def collate_fn(batch):
    # Find the maximum time dimension in the batch
    max_len = max([item.shape[2] for item in batch])
    
    # Pad all items in the batch to the max length
    padded_batch = [F.pad(item, (0, max_len - item.shape[2]), "constant", 0) for item in batch]
    
    # Stack the padded tensors into a single tensor
    return torch.stack(padded_batch)

class DrumDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load and process the audio
        signal, sr = librosa.load(self.file_paths[idx], sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalize the spectrogram
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

        # Add channel dimension for CNN
        mel_spectrogram = mel_spectrogram.unsqueeze(0)

        return mel_spectrogram

file_paths = [str(p) for p in Path('data/Kicks').glob('*.wav')]

dataset = DrumDataset(file_paths)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

class DrumCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Assuming input Mel spectrogram is (1, 128, 128)
        self.fc2 = nn.Linear(128, 64)  # Output feature embedding size

    def forward(self, x):
        # Pass input through convolutional layers with ReLU activation and pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # Output size will be (16, 64, 64)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # Output size will be (32, 32, 32)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)  # Output size will be (64, 16, 16)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * 16 * 16)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # This will be the feature embedding
        
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Define the model, loss, and optimizer
model = DrumCNN()
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop
for epoch in range(10):  # Number of epochs
    for batch in dataloader:
        # Assume `batch` contains positive and negative pairs
        sample1, sample2, label = batch  # `label` is 0 for similar, 1 for dissimilar

        # Forward pass
        output1 = model(sample1)
        output2 = model(sample2)

        # Compute loss
        loss = criterion(output1, output2, label)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
        

# # Extract features for two samples
# sample1, sr1 = librosa.load('path/to/sample1.wav', sr=None)
# sample2, sr2 = librosa.load('path/to/sample2.wav', sr=None)

# # Convert to spectrograms and get features
# spectrogram1 = dataset[0].unsqueeze(0)  # Add batch dimension
# spectrogram2 = dataset[1].unsqueeze(0)

# with torch.no_grad():
#     feature1 = model(spectrogram1)
#     feature2 = model(spectrogram2)

# # Compute similarity
# similarity = cosine_similarity(feature1.numpy(), feature2.numpy())
# print(f'Similarity Score: {similarity}')