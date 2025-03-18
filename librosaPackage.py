import librosa
import matplotlib.pyplot as plt
import librosa.display
import os

directory = os.getcwd()
filename = '/loudly.mp3'
filepath = directory + filename

# Load the audio file
y, sr = librosa.load(filepath, sr=None)  

# If the audio has multiple channels (e.g., stereo), convert to mono
if y.ndim > 1:
    y = librosa.to_mono(y)

print(f"Audio loaded. Duration = {len(y)/sr:.2f} seconds, Sampling Rate = {sr} Hz")

tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time', hop_length=512, tightness=100)

print("Type of tempo variable:", type(tempo))
print("Value of tempo:", tempo)
print("Beat times (in seconds):", beat_times)

# Plot waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.5) 


plt.vlines(beat_times, ymin=-0.5, ymax=0.5, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
plt.title("Waveform with Detected Beats")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.xlim([0, max(beat_times) + 1]) 

plt.legend()
plt.tight_layout()
plt.show()