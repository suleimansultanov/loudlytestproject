import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader

class BeatDataset(Dataset):
    def __init__(self, audio_files, beat_annotations, sr=22050, hop_length=512):
        self.audio_files = audio_files
        self.beat_annotations = beat_annotations
        self.sr = sr
        self.hop_length = hop_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load audio file
        y, sr = librosa.load(self.audio_files[idx], sr=self.sr)

        # Compute log Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=self.hop_length, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec)

        # Normalize
        log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)

        # Load beat annotations
        beat_times = np.loadtxt(self.beat_annotations[idx])

        # Convert beat times to beat activation labels
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=self.hop_length)
        labels = np.zeros(log_mel_spec.shape[1])
        labels[beat_frames] = 1  # Mark beats with 1

        return torch.tensor(log_mel_spec, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


class BeatRNN(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, num_layers=2):
        super(BeatRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # Bidirectional LSTM
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for LSTM (batch, seq, features)
        out, _ = self.rnn(x)
        out = self.fc(out).squeeze(-1)
        return self.sigmoid(out)


def train_model(audio_files, beat_annotations, num_epochs=20, batch_size=16, lr=0.001):
    dataset = BeatDataset(audio_files, beat_annotations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BeatRNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # Binary cross-entropy loss for beat detection

    for epoch in range(num_epochs):
        epoch_loss = 0
        for spectrograms, labels in dataloader:
            optimizer.zero_grad()
            predictions = model(spectrograms)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

    return model

directory = os.getcwd()
audio_dir = os.path.join(directory,'dataset/audio')
beats_dir = os.path.join(directory,'dataset/annotations')

audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))

beat_annotations = sorted(glob.glob(os.path.join(beats_dir, '*.beats')))

# Train the model
trained_model = train_model(audio_files, beat_annotations)
torch.save(trained_model, "beat_madmom_model.pth")
