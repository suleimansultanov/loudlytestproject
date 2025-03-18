import madmom
import numpy as np
import plotly.graph_objects as go
import os

directory = os.getcwd()
filename = '/loudly.mp3'
filepath = directory + filename

processor = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()(filepath)
beats = processor(act)

if len(beats) > 1:
    intervals = np.diff(beats)
    avg_interval = np.mean(intervals)
    bpm = 60.0 / avg_interval
else:
    bpm = 0

print("Detected beats (in seconds):", beats)
print(f"Estimated BPM: {bpm:.2f}")

signal = madmom.audio.signal.Signal(filepath, num_channels=1, sample_rate=44100)
sample_rate = signal.sample_rate
time_axis = np.arange(len(signal)) / sample_rate

fig = go.Figure()

fig.add_trace(go.Scatter(x=time_axis, y=signal, mode='lines', name='Audio Waveform'))

for beat in beats:
    fig.add_trace(go.Scatter(x=[beat, beat], y=[min(signal), max(signal)], mode='lines', line=dict(color='red', dash='dash'), name='Beat'))

fig.update_layout(title=f'Audio Waveform and Detected Beats (BPM: {bpm:.2f})',
                  xaxis_title='Time (seconds)', yaxis_title='Amplitude',
                  template="plotly_white")

fig.show()
