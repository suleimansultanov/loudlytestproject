import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os


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

# Load the trained model (ensure "beat_rnn_weights.pth" is saved from training)
path = os.path.join(os.getcwd(), "models/beat_madmom_model.pth")
model = BeatRNN()
torch.load(path,weights_only=False)
model.eval()  # Set to evaluation mode