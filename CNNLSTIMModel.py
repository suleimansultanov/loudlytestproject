import matplotlib.pyplot as plt
import librosa
import numpy as np
import glob
import os
from tensorflow.keras.losses import BinaryFocalCrossentropy
import scipy.signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D, BatchNormalization

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

    # Загрузка оригинального аудио
    y, sr = librosa.load(audio_path, sr=22050, duration=30)

    # **Оригинальный пример**
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=4000)
    S_DB = librosa.power_to_db(S, ref=np.max)
    X_original = np.expand_dims(S_DB.T, axis=-1)
    X_train.append(X_original)

    # Загрузка битов
    beat_times = np.loadtxt(beat_path)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    y_label = np.zeros((X_original.shape[0], 1))
    beat_frames = beat_frames[beat_frames < len(y_label)]
    y_label[beat_frames] = 1
    y_train.append(y_label)

    # **Augmentation: добавляем шум**
    y_noise = y + 0.005 * np.random.randn(len(y))
    S_noise = librosa.feature.melspectrogram(y=y_noise, sr=sr, n_mels=64, fmax=4000)
    S_DB_noise = librosa.power_to_db(S_noise, ref=np.max)
    X_noise = np.expand_dims(S_DB_noise.T, axis=-1)
    X_train.append(X_noise)
    y_train.append(y_label)  # Метки те же самые

    # **Augmentation: изменение скорости (ускорение)**
    y_fast = librosa.effects.time_stretch(y, rate=1.1)
    if len(y_fast) >= sr * 30:  # убедимся, что длина не меньше 30 сек.
        y_fast = y_fast[:sr*30]
    else:
        y_fast = np.pad(y_fast, (0, max(0, sr*30 - len(y_fast))), mode='constant')

    S_fast = librosa.feature.melspectrogram(y=y_fast, sr=sr, n_mels=64, fmax=4000)
    S_DB_fast = librosa.power_to_db(S_fast, ref=np.max)
    X_fast = np.expand_dims(S_DB_fast.T, axis=-1)
    X_train.append(X_fast)
    y_train.append(y_label)  # Метки те же самые

# Объединяем данные после аугментации
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

print("Данные подготовлены (с augmentation):", X_train.shape, y_train.shape)


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

# Классовые веса, учитывающие дисбаланс (подбери под себя)
class_weights = {0: 0.1, 1: 0.9}

model.compile(optimizer='adam', 
              loss=BinaryFocalCrossentropy(gamma=2.0), 
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=42, batch_size=16, class_weight=class_weights)
# Обучение

audio_path = 'loudly.mp3'
y_test, sr_test = librosa.load(audio_path, sr=22050, duration=30)

S_test = librosa.feature.melspectrogram(y=y_test, sr=sr_test, n_mels=64, fmax=4000)
S_DB_test = librosa.power_to_db(S_test, ref=np.max)

X_test = np.expand_dims(S_DB_test.T, -1)

# предсказываем удары
# Предсказанные вероятности битов
predicted_beats = model.predict(X_test).flatten()

# Сглаживание (сильнее медианный фильтр)
smoothed_beats = scipy.signal.medfilt(predicted_beats, kernel_size=9)

# Адаптивный порог
beat_threshold = np.mean(smoothed_beats) + np.std(smoothed_beats) * 0.5
beat_frames = np.where(smoothed_beats > beat_threshold)[0]

beat_times = librosa.frames_to_time(beat_frames, sr=sr_test)

# Убираем слишком частые и слишком редкие пики (улучшение стабильности)
min_interval = 0.3  # 200 мс между битами (~300 BPM максимум)
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

# 🔹 Отображаем волну аудиофайла
librosa.display.waveshow(y_test, sr=sr_test, alpha=0.6, color='b')

plt.title("Waveform with Detected Beats")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.ylim(-1, 1)  # Ограничиваем амплитуду
plt.show()