import librosa
import matplotlib.pyplot as plt
import librosa.display

# Load the audio file (preserve native sampling rate by setting sr=None)
y, sr = librosa.load('loudly.mp3', sr=None)  

# If the audio has multiple channels (e.g., stereo), convert to mono
if y.ndim > 1:
    y = librosa.to_mono(y)

print(f"Audio loaded. Duration = {len(y)/sr:.2f} seconds, Sampling Rate = {sr} Hz")



# Perform beat tracking
tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time')

# Output the results
print("Type of tempo variable:", type(tempo))
print("Value of tempo:", tempo)
print("Beat times (in seconds):", beat_times)



# Plot waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.5)  # plot waveform amplitude over time

# Overlay beat markers as vertical lines
#for bt in beat_times:
#    plt.axvline(x=bt, color='r', linestyle='--', label='Beat' if bt == beat_times[0] else None)

for beat_time in beat_times:
    plt.axvline(x=beat_time, color='red', linestyle='--', alpha=0.5, linewidth=0.5)  # Adjust alpha and linewidth



plt.title("Waveform with Detected Beats")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.xlim([0, max(beat_times) + 1])  # Adjust x-axis to cover slightly more than the last beat time

plt.legend()
plt.tight_layout()
plt.show()

