import madmom
import numpy as np
import plotly.graph_objects as go
import os

# File path setup
directory = os.getcwd()
filename = 'loudly.mp3'
filepath = os.path.join(directory, filename)

# Load audio and process beats
processor = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()(filepath)
beats = processor(act)

# Define segment length (e.g., 10 seconds per segment)
segment_length = 10  # seconds
bpm_over_time = []
time_stamps = []

if len(beats) > 1:
    for i in range(0, int(beats[-1]), segment_length):
        # Get beats in the current segment
        segment_beats = beats[(beats >= i) & (beats < i + segment_length)]
        
        if len(segment_beats) > 1:
            intervals = np.diff(segment_beats)
            avg_interval = np.mean(intervals)
            segment_bpm = 60.0 / avg_interval
        else:
            segment_bpm = 0  # No beats detected
        
        bpm_over_time.append(segment_bpm)
        time_stamps.append(i)
    
    avg_bpm = np.mean([bpm for bpm in bpm_over_time if bpm > 0])  # Average BPM ignoring zeros
else:
    avg_bpm = 0

# Print detected BPMs
print("Detected beats (in seconds):", beats)
print(f"Overall Estimated BPM: {avg_bpm:.2f}")
print("BPM per segment:", list(zip(time_stamps, bpm_over_time)))

# Load audio for visualization
signal = madmom.audio.signal.Signal(filepath, num_channels=1, sample_rate=44100)
sample_rate = signal.sample_rate
time_axis = np.arange(len(signal)) / sample_rate

fig = go.Figure()

# Plot audio waveform
fig.add_trace(go.Scatter(x=time_axis, y=signal, mode='lines', name='Audio Waveform'))

# Plot detected beats
for beat in beats:
    fig.add_trace(go.Scatter(x=[beat, beat], y=[min(signal), max(signal)], mode='lines', line=dict(color='red', dash='dash'), name='Beat'))

# Plot BPM variations over time
fig.add_trace(go.Scatter(x=time_stamps, y=bpm_over_time, mode='lines+markers', name='BPM per Segment', line=dict(color='blue')))

fig.update_layout(title=f'Audio Waveform and BPM Analysis (Avg BPM: {avg_bpm:.2f})',
                  xaxis_title='Time (seconds)', yaxis_title='Amplitude/BPM',
                  template="plotly_white")

fig.show()
