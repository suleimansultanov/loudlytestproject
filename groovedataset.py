import tensorflow_datasets as tfds
import pretty_midi
import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D
import io

# Загрузка Groove MIDI dataset (маленький кусок)
dataset, info = tfds.load('groove', split='train[:20]', with_info=True)

X_train, y_train = [], []

for example in tfds.as_numpy(dataset):
    midi_bytes = example['midi']  # Это байты MIDI-файла
    midi_data = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))

    # Генерируем аудио из MIDI
    audio = midi_data.fluidsynth(fs=22050)

    # Извлекаем мел-спектрограмму
    S = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=64, fmax=4000)
    S_DB = librosa.power_to_db(S, ref=np.max)

    X = np.expand_dims(S_DB.T, axis=-1)
    X_train.append(X)

    # Метки ударов из MIDI
    beat_times = midi_data.get_beats()
    beat_frames = librosa.time_to_frames(beat_times, sr=22050)

    y_label = np.zeros((X.shape[0], 1))
    beat_frames = beat_frames[beat_frames < len(y_label)]
    y_label[beat_frames] = 1
    y_train.append(y_label)

# Конкатенация данных
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

print("Данные подготовлены:", X_train.shape, y_train.shape)



model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    LSTM(64, return_sequences=True),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# обучение
class_weights = {0: 0.1, 1: 0.9}
model.fit(X_train, y_train, epochs=5, batch_size=16, class_weight=class_weights)


audio_path = 'loudly.mp3'
y_test, sr_test = librosa.load(audio_path, sr=22050, duration=30)

S_test = librosa.feature.melspectrogram(y=y_test, sr=sr_test, n_mels=64, fmax=4000)
S_DB_test = librosa.power_to_db(S_test, ref=np.max)

X_test = np.expand_dims(S_DB_test.T, -1)

# предсказываем удары
beat_threshold = 0.2
predicted_beats = model.predict(X_test)
beat_frames = np.where(predicted_beats.flatten() > beat_threshold)[0]

beat_times = librosa.frames_to_time(beat_frames, sr=sr_test)



if len(beat_times) > 1:
    bpm = 60 / np.mean(np.diff(beat_times))
    print(f"Calculated BPM: {bpm:.2f}")
else:
    print("Not enough beats detected to calculate BPM.")



