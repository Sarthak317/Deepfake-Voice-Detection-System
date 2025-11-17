import torch
import librosa
import numpy as np
import os

# Load Trained Model
class AudioClassifier(torch.nn.Module):
    def __init__(self, input_size=13):
        super(AudioClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Function to Predict Real/Fake
def predict_audio(file_path, model_path="audio_model.pth"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = AudioClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    try:
        # Load audio with 16kHz sample rate and mono
        audio, _ = librosa.load(file_path, sr=16000, mono=True)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    tensor = torch.tensor(mfcc_mean, dtype=torch.float32).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, 1).item()

    return "Fake" if prediction == 0 else "Real"

# Run Detection
if __name__ == "__main__":
    # Use raw string to avoid backslash issues
    file_path = r"C:\Users\sarth\OneDrive\Desktop\Fake Voice Detection System\data\real"
    
    try:
        result = predict_audio(file_path)
        print(f"Prediction: {result}")
    except Exception as e:
        print(f"Error: {e}")
