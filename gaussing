import tensorflow as tf
import librosa
import numpy as np
import glob
import os
import scipy.signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.losses import BinaryFocalCrossentropy
import matplotlib.pyplot as plt
import librosa.display

X_train, y_train = [], []

audio_dir = 'datasethiphop/audio'
beats_dir = 'datasethiphop/annotations'

audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))

for audio_path in audio_files:
    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    beat_path = os.path.join(beats_dir, f'{file_id}.beats')

    if not os.path.exists(beat_path):
        print(f"Аннотация не найдена для {audio_path}, пропускаем.")
        continue

    # Загрузка аудио
    y, sr = librosa.load(audio_path, sr=22050, duration=30)

    # Mel-спектрограмма
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=4000)
    S_DB = librosa.power_to_db(S, ref=np.max)
    X_original = np.expand_dims(S_DB.T, axis=-1)

    # Загрузка битов из .beats-файла
    beat_times = np.loadtxt(beat_path)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)

    # ---- Улучшение меток методом Gaussian smoothing (важный шаг!) ----
    y_label = np.zeros(X_original.shape[0])
    beat_frames = beat_frames[beat_frames < len(y_label)]
    y_label[beat_frames] = 1

    # Gaussian smoothing
    gaussian_kernel = scipy.signal.gaussian(9, std=1)
    y_label_smoothed = np.convolve(y_label, gaussian_kernel, mode='same')
    y_label_smoothed = np.clip(y_label_smoothed, 0, 1)
    y_label_smoothed = np.expand_dims(y_label_smoothed, axis=-1)

    X_train.append(X_original)
    y_train.append(y_label_smoothed.reshape(-1, 1))

    # Augmentation: добавляем шум
    y_noise = y + 0.005 * np.random.randn(len(y))
    S_noise = librosa.feature.melspectrogram(y=y_noise, sr=sr, n_mels=64, fmax=4000)
    S_DB_noise = librosa.power_to_db(S_noise, ref=np.max)
    X_noise = np.expand_dims(S_DB_noise.T, axis=-1)
    X_train.append(X_noise)
    y_train.append(y_label_smoothed)

    # Augmentation: изменение скорости
    y_fast = librosa.effects.time_stretch(y, rate=1.1)
    if len(y_fast) >= sr * 30:
        y_fast = y_fast[:sr*30]
    else:
        y_fast = np.pad(y_fast, (0, sr*30 - len(y_fast)), mode='constant')

    S_fast = librosa.feature.melspectrogram(y=y_fast, sr=sr, n_mels=64, fmax=4000)
    S_DB_fast = librosa.power_to_db(S_fast, ref=np.max)
    X_fast = np.expand_dims(S_DB_fast.T, axis=-1)
    X_train.append(X_fast)
    y_train.append(y_label_smoothed)

# Объединяем данные
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

print("Данные подготовлены (с Gaussian smoothing):", X_train.shape, y_train.shape)

# ------------------
# Модель
# ------------------
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0), 
              metrics=['accuracy'])

class_weights = {0: 0.05, 1: 0.95}
model.fit(X_train, y_train, epochs=20, batch_size=16, class_weight=class_weights)

# ------------------
# Квантование модели
# ------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('beat_detection_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
print("Quantized model saved successfully!")

# ------------------
# Тестирование
# ------------------
audio_path = 'loudly.mp3'
y_test, sr_test = librosa.load(audio_path, sr=22050, duration=30)

S_test = librosa.feature.melspectrogram(y=y_test, sr=sr_test, n_mels=64, fmax=4000)
S_DB_test = librosa.power_to_db(S_test, ref=np.max)
X_test = np.expand_dims(S_DB_test.T, -1)

predicted_beats = model.predict(X_test).flatten()

# Сглаживание и адаптивный порог
smoothed_beats = scipy.signal.medfilt(predicted_beats, kernel_size=9)
beat_threshold = np.mean(smoothed_beats) + 0.5 * np.std(smoothed_beats)
beat_frames = np.where(smoothed_beats > beat_threshold)[0]

beat_times = librosa.frames_to_time(beat_frames, sr=sr_test)

if len(beat_times) > 1:
    bpm = 60 / np.mean(np.diff(beat_times))
    print(f"Calculated BPM: {bpm:.2f}")
else:
    print("Not enough beats detected to calculate BPM.")

# ------------------
# Визуализация результатов
# ------------------
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y_test, sr=sr_test, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Detected Beats')
plt.title(f"Audio Waveform with Beats (BPM: {bpm:.2f})")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
