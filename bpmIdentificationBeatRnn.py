import torch
import librosa
import numpy as np
import scipy.signal as signal
import os
import matplotlib.pyplot as plt
from models.beat_rnn import BeatRNN
# Load the trained model
model = BeatRNN()  # Ensure this matches the trained model architecture
torch.load(os.path.join(os.getcwd(),"models/beat_madmom_model.pth"), weights_only=False)
model.eval()  #
# Function to compute BPM per segment
def compute_bpm_per_segment(audio_file, segment_length=5.0, sr=22050, hop_length=512):

    # Load audio file
    y, sr = librosa.load(audio_file, sr=sr)

    # Compute log Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec)

    # Normalize
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)

    # Split into segments
    segment_frames = int(segment_length * sr / hop_length)  # Number of frames per segment
    num_segments = log_mel_spec.shape[1] // segment_frames
    bpm_values = []
    segment_times = []

    for i in range(num_segments):
        # Extract segment
        segment = log_mel_spec[:, i * segment_frames: (i + 1) * segment_frames]
        if segment.shape[1] == 0:
            continue
        
        # Convert to tensor and reshape
        input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 128, time_steps)

        # Predict beats
        with torch.no_grad():
            beat_predictions = model(input_tensor).numpy().flatten()

        # Convert predictions to beat times
        beat_times = librosa.frames_to_time(np.where(beat_predictions > 0.5)[0], sr=sr, hop_length=hop_length)

        # Compute BPM if we have detected beats
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            bpm_per_segment = 60.0 / beat_intervals
            bpm_values.append(np.median(bpm_per_segment))  # Use median BPM per segment
        else:
            bpm_values.append(0)  # No beats detected in segment

        # Store segment center time
        segment_times.append((i + 0.5) * segment_length)

    # Apply median filtering for smoother BPM estimates
    smoothed_bpm = signal.medfilt(bpm_values, kernel_size=3)

    return smoothed_bpm, segment_times

# ===========================
#  Example Usage
# ===========================
audio_file = "loudly.mp3"  # Replace with your audio file
segment_length = 5.0  # Compute BPM per 5-second segment

bpm_values, segment_times = compute_bpm_per_segment(audio_file, segment_length=segment_length)
valid_bpm_values = [bpm for bpm in bpm_values if bpm > 0]


# ===========================
#  Visualization
# ===========================
plt.figure(figsize=(10, 4))
plt.plot(segment_times, bpm_values, label="Estimated BPM per Segment", marker="o", linestyle="dashed", color="blue")
plt.xlabel("Time (seconds)")
plt.ylabel("BPM")
plt.title("BPM Estimation for Each Segment")
plt.legend()
plt.show()

