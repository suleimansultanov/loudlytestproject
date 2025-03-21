import madmom
import numpy as np
import plotly.graph_objects as go
import os
import pickle  

with open('madmom_genre_classifier.pkl', 'rb') as model_file:
    genre_model = pickle.load(model_file)


GENRE_LABELS = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']


directory = os.getcwd()
filename = 'test1.mp3'
filepath = os.path.join(directory, filename)


def extract_madmom_features(filepath):

    processor = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(filepath)
    beats = processor(act)


    tempo_processor = madmom.features.tempo.TempoEstimationProcessor(fps=100)
    tempo = tempo_processor(act)

   
    onset_env = madmom.features.onsets.RNNOnsetProcessor()(filepath)


    features = [
        np.mean(tempo[:, 0]), 
        np.std(tempo[:, 0]),   
        np.mean(onset_env),    
        np.std(onset_env),     
        len(beats) / (beats[-1] - beats[0] if len(beats) > 1 else 1),  # Beat density
    ]
    return np.array(features).reshape(1, -1)


features = extract_madmom_features(filepath)
predicted_genre = genre_model.predict(features)[0]


processor = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()(filepath)
beats = processor(act)


segment_length = 10  
bpm_over_time = []
time_stamps = []

if len(beats) > 1:
    for i in range(0, int(beats[-1]), segment_length):
        segment_beats = beats[(beats >= i) & (beats < i + segment_length)]
        
        if len(segment_beats) > 1:
            intervals = np.diff(segment_beats)
            avg_interval = np.mean(intervals)
            segment_bpm = 60.0 / avg_interval
        else:
            segment_bpm = 0  # No beats detected
        
        bpm_over_time.append(segment_bpm)
        time_stamps.append(i)
    
    avg_bpm = np.mean([bpm for bpm in bpm_over_time if bpm > 0])  
else:
    avg_bpm = 0


print("Detected beats (in seconds):", beats)
print(f"Overall Estimated BPM: {avg_bpm:.2f}")
print("BPM per segment:", list(zip(time_stamps, bpm_over_time)))
print(f"Predicted Genre: {GENRE_LABELS[predicted_genre]}")


signal = madmom.audio.signal.Signal(filepath, num_channels=1, sample_rate=44100)
sample_rate = signal.sample_rate
time_axis = np.arange(len(signal)) / sample_rate

fig = go.Figure()


fig.add_trace(go.Scatter(x=time_axis, y=signal, mode='lines', name='Audio Waveform'))


for beat in beats:
    fig.add_trace(go.Scatter(x=[beat, beat], y=[min(signal), max(signal)], mode='lines', line=dict(color='red', dash='dash'), name='Beat'))


fig.add_trace(go.Scatter(x=time_stamps, y=bpm_over_time, mode='lines+markers', name='BPM per Segment', line=dict(color='blue')))

fig.update_layout(title=f'Audio Waveform, BPM & Genre: {GENRE_LABELS[predicted_genre]} (Avg BPM: {avg_bpm:.2f})',
                  xaxis_title='Time (seconds)', yaxis_title='Amplitude/BPM',
                  template="plotly_white")

fig.show()
