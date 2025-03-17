import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D
import librosa.display
import matplotlib.pyplot as plt

# --- Загрузка и подготовка аудио из твоего собственного файла ---
audio_path = 'loudly.mp3'
y, sr = librosa.load(audio_path, sr=22050, duration=10)  # первые 10 секунд

# Извлекаем мел-спектрограмму
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=4000)
S_DB = librosa.power_to_db(S, ref=np.max)

X_train = np.expand_dims(S_DB.T, axis=-1)

# Искусственные метки для демонстрации:
y_train = np.zeros((X_train.shape[0], 1))
y_train[::10] = 1  # условные "удары" каждые 10 кадров

# --- Простая и быстрая модель для демонстрации ---
model = Sequential([
    Conv1D(16, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    LSTM(32, return_sequences=True),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Быстрая тренировка:
model.fit(X_train, y_train, epochs=3, batch_size=8)

# --- Предсказание ударов и расчёт BPM ---
predicted_beats = model.predict(X_train)

beat_threshold = 0.5
beat_frames = np.where(predicted_beats.flatten() > beat_threshold)[0]
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

if len(beat_times) > 1:
    beat_intervals = np.diff(beat_times)
    bpm = 60 / np.mean(beat_intervals)
    print(f"Estimated BPM: {bpm:.2f}")
else:
    print("Not enough beats detected to calculate BPM.")

# --- Визуализация ---
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r', alpha=0.75, linestyle='--', label='Detected Beats')
plt.title('Waveform with Detected Beats')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
