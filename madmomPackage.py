import madmom

# Specify the path to your audio file
audio_file = 'loudly.mp3'

# Set up the beat tracking processor
# DBNBeatTracker uses a neural network combined with dynamic Bayesian networks
processor = madmom.features.beats.DBNBeatTracker(fps=100)

# Process the audio file to get the beats
beats = processor(audio_file)

# Print the detected beats
print("Detected beats (in seconds):", beats)

# Optional: Plot the results if you want a visual representation
import matplotlib.pyplot as plt

# Load the audio file to display the waveform
signal, sample_rate = madmom.audio.signal.load_wave(audio_file, sample_rate=None)

# Plotting the waveform and the beats
plt.figure(figsize=(10, 4))
plt.plot(signal)
for beat in beats:
    plt.axvline(x=beat * sample_rate, color='red', linestyle='--')
plt.title('Audio Waveform and Detected Beats')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()
