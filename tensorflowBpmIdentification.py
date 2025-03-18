import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
import scipy.signal
import tensorflow as tf

model_fileName = "cnn_lstm_bpm_model.h5"
directory = os.getcwd()
model_path = os.path.join(directory, model_fileName)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found! Train or provide the .h5 file.")

model = tf.keras.models.load_model(model_path)

filename = 'loudly.mp3'
audio_path = os.path.join(directory, filename)

y_test, sr_test = librosa.load(audio_path, sr=22050, duration=30)

S_test = librosa.feature.melspectrogram(y=y_test, sr=sr_test, n_mels=64, fmax=4000)
S_DB_test = librosa.power_to_db(S_test, ref=np.max)

X_test = np.expand_dims(S_DB_test.T, -1)

# predict beats
predicted_beats = model.predict(X_test).flatten()

# Med filter
smoothed_beats = scipy.signal.medfilt(predicted_beats, kernel_size=9)

# treshhold 
beat_threshold = np.mean(smoothed_beats) + np.std(smoothed_beats) * 0.5
beat_frames = np.where(smoothed_beats > beat_threshold)[0]

beat_times = librosa.frames_to_time(beat_frames, sr=sr_test)

# add min interval 
min_interval = 0.3 
final_beats = [beat_times[0]] if beat_times.size else []

for time in beat_times[1:]:
    if time - final_beats[-1] > min_interval:
        final_beats.append(time)

final_beats = np.array(final_beats)


if len(final_beats) > 1:
    bpm = 60 / np.mean(np.diff(final_beats))
    print(f"Calculated BPM: {bpm:.2f}")
else:
    print("Not enough beats detected to calculate BPM.")

plt.figure(figsize=(12, 4))

# Display plot
librosa.display.waveshow(y_test, sr=sr_test, alpha=0.6, color='b')

plt.title("Waveform with Detected Beats")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.ylim(-1, 1)  
plt.show()