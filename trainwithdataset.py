import tensorflow as tf
import tensorflow_datasets as tfds
import pretty_midi
import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D
import librosa
import numpy as np
import glob
import os
from tensorflow.keras.losses import BinaryFocalCrossentropy

X_train, y_train = [], []

audio_dir = 'datasethiphop/audio'
beats_dir = 'datasethiphop/annotations'

audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))

for audio_path in audio_files:
    # Получаем имя файла без расширения (например, "1", "2")
    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    beat_path = os.path.join(beats_dir, f'{file_id}.beats')

    if not os.path.exists(beat_path):
        print(f"Аннотация не найдена для {audio_path}, пропускаем.")
        continue

    # Загрузка аудио
    y, sr = librosa.load(audio_path, sr=22050, duration=30)

    # Мел-спектрограмма
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=4000)
    S_DB = librosa.power_to_db(S, ref=np.max)

    X = np.expand_dims(S_DB.T, axis=-1)
    X_train.append(X)

    # Загрузка битов из .beats-файла
    beat_times = np.loadtxt(beat_path)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)

    y_label = np.zeros((X.shape[0], 1))
    beat_frames = beat_frames[beat_frames < len(y_label)]
    y_label[beat_frames] = 1

    y_train.append(y_label)

# Объединяем данные
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

print("Данные успешно подготовлены:", X_train.shape, y_train.shape)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D, BatchNormalization

model = Sequential([
    # Первый сверточный блок
    Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(2),

    # Второй сверточный блок
    Conv1D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    # LSTM-блок (анализ последовательностей)
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    
    # Дополнительный рекуррентный слой
    LSTM(64, return_sequences=True),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),

    # Финальный слой бинарной классификации (бит или не бит)
    Dense(1, activation='sigmoid')
])

# Компиляция модели с оптимизатором Adam
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
              metrics=['accuracy'])

# Классовые веса, учитывающие дисбаланс (подбери под себя)
class_weights = {0: 0.1, 1: 0.9}

# Обучение
model.fit(X_train, y_train, epochs=20, batch_size=16, class_weight=class_weights)


audio_path = 'loudly.mp3'
y_test, sr_test = librosa.load(audio_path, sr=22050, duration=30)

S_test = librosa.feature.melspectrogram(y=y_test, sr=sr_test, n_mels=64, fmax=4000)
S_DB_test = librosa.power_to_db(S_test, ref=np.max)

X_test = np.expand_dims(S_DB_test.T, -1)

# предсказываем удары
predicted_beats = model.predict(X_test)
beat_threshold = 0.5

# Используем медианную фильтрацию для удаления шума
import scipy.signal
filtered_beats = scipy.signal.medfilt(predicted_beats.flatten(), kernel_size=5)

beat_frames = np.where(filtered_beats > beat_threshold)[0]
beat_times = librosa.frames_to_time(beat_frames, sr=sr_test)

if len(beat_times) > 1:
    bpm = 60 / np.mean(np.diff(beat_times))
    
    print(f"Calculated BPM: {bpm:.2f}")
else:
    print("Not enough beats detected to calculate BPM.")


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y_test, sr=sr_test, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r', linestyle='--')
plt.show()
