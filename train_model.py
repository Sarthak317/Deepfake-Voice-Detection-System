import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import random

# Ensure data folders exist
os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

# Custom Dataset Class
class AudioDataset(Dataset):
    def __init__(self, real_dir, fake_dir, sr=16000, n_mfcc=13):
        self.real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".wav")]
        self.fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".wav")]
        self.all_files = [(f, 1) for f in self.real_files] + [(f, 0) for f in self.fake_files]
        
        if not self.all_files:
            raise ValueError("No audio files found! Please add .wav files to 'data/real' and 'data/fake'.")
        
        random.shuffle(self.all_files)
        self.sr = sr
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path, label = self.all_files[idx]
        audio, _ = librosa.load(file_path, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc = np.mean(mfcc, axis=1)  # Average over time
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Neural Network Model
class AudioClassifier(nn.Module):
    def __init__(self, input_size=13):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)  # Binary classification (Real/Fake)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Training Function
def train_model(real_dir, fake_dir, epochs=10, batch_size=8, lr=0.001):
    try:
        dataset = AudioDataset(real_dir, fake_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioClassifier()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "audio_model.pth")
    print("✅ Model trained and saved as 'audio_model.pth'!")

# Run Training
if __name__ == "__main__":
    train_model("data/real", "data/fake")
