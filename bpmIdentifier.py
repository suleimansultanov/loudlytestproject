import librosa
import matplotlib.pyplot as plt
import librosa.display
import plotly.graph_objs as go

#print("Current working directory:", os.getcwd())
filepath = 'test1.mp3'

# Load the audio file (preserve native sampling rate by setting sr=None)
y, sr = librosa.load(filepath, sr=None)  

# Downsample to 22050 Hz if higher resolution is unnecessary
#y, sr = librosa.load(filepath, sr=22050) 

#If only certain parts of the audio are of interest, consider loading or processing segments rather than the entire track.
#duration = 30  # seconds
#y, sr = librosa.load(filepath, duration=duration)


# If the audio has multiple channels (e.g., stereo), convert to mono
if y.ndim > 1:
    y = librosa.to_mono(y)

print(f"Audio loaded. Duration = {len(y)/sr:.2f} seconds, Sampling Rate = {sr} Hz")

#Apply noise reduction or a bandpass filter to the audio to enhance beat detection, especially in noisy recordings.
#b, a = scipy.signal.butter(3, [0.2, 0.5], btype='band')
#y_filtered = scipy.signal.filtfilt(b, a, y)

# Perform beat tracking
tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time', hop_length=512, tightness=100)



# Output the results
print("Type of tempo variable:", type(tempo))
print("Value of tempo:", tempo)
print("Beat times (in seconds):", beat_times)



# Plot waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.5)  # plot waveform amplitude over time

#First aproach for plot visualization


# Overlay beat markers as vertical lines
#for bt in beat_times:
#    plt.axvline(x=bt, color='r', linestyle='--', label='Beat' if bt == beat_times[0] else None)


# for beat_time in beat_times:
#     plt.axvline(x=beat_time, color='red', linestyle='--', alpha=0.5, linewidth=0.5)  # Adjust alpha and linewidth


# plt.title("Waveform with Detected Beats")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Amplitude")

# plt.xlim([0, max(beat_times) + 1])  # Adjust x-axis to cover slightly more than the last beat time

# plt.legend()
# plt.tight_layout()
# plt.show()


# Second aproach for plot visualization
# Prepare data for Plotly
trace_waveform = go.Scatter(
    x = [x/sr for x in range(len(y))],  # Convert sample indices to time in seconds
    y = y,
    mode = 'lines',
    name = 'Waveform'
)

# Adding beat markers as vertical lines
trace_beats = go.Scatter(
    x = beat_times,
    y = [0] * len(beat_times),  # Y positions for markers, adjust if necessary
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'red',
        symbol = 'line-ns-open'
    ),
    name = 'Beats'
)

# Define layout
layout = go.Layout(
    title = 'Waveform and Beat Detection',
    xaxis = dict(title = 'Time (seconds)'),
    yaxis = dict(title = 'Amplitude'),
    showlegend = True
)

# Combine traces
fig = go.Figure(data=[trace_waveform, trace_beats], layout=layout)

# Show plot
fig.show()
