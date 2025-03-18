import os
import numpy as np
import madmom
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Reshape
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, Conv1D, MaxPooling1D, BatchNormalization


dataset_path = "dataset"
audio_path = os.path.join(dataset_path, "audio")
annotations_path = os.path.join(dataset_path, "annotations")


def calculate_bpm(beats_file):
    try:
        beats = np.loadtxt(beats_file)  
        if len(beats) < 2:
            return 0 
        
        intervals = np.diff(beats)  
        avg_interval = np.mean(intervals)  
        bpm = 60.0 / avg_interval  
        
        return bpm
    except Exception as e:
        print(f"Error reading {beats_file}: {e}")
        return 0  


def extract_spectrogram(filepath):
    spectrogram_processor = madmom.audio.spectrogram.SpectrogramProcessor(frame_size=2048, fps=100)
    spectrogram = spectrogram_processor(filepath)
    return spectrogram


X = []
y = []

# Iterate over audio files
for file in os.listdir(audio_path):
    if file.endswith(".wav"):  
        file_name = os.path.splitext(file)[0]  # Get filename without extension
        audio_file = os.path.join(audio_path, file)
        beats_file = os.path.join(annotations_path, f"{file_name}.beats")

        if not os.path.exists(beats_file):
            print(f"Skipping {file}, missing beats file: {beats_file}")
            continue

        # Get BPM from beats file
        bpm = calculate_bpm(beats_file)
        if bpm == 0:
            print(f"Skipping {file}, BPM could not be determined.")
            continue

        try:
            # Extract spectrogram
            spectrogram = extract_spectrogram(audio_file)

            # Ensure spectrogram has correct 4D shape for CNN input**
            spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension (height, width, 1)

            X.append(spectrogram)
            y.append(bpm)
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# **Fix: Ensure the correct input shape for CNN**
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # (batch_size, height, width, channels)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adjust CNN-LSTM model to match input shape**
def build_model():
    model = Sequential([
        # First CNN layer
        Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(2),

        # Second CNN layer
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),

        # LSTM layer
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        
        # Additional layer
        LSTM(64, return_sequences=True),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),

        # Final binary
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', 
              loss=BinaryFocalCrossentropy(gamma=2.0),
              metrics=['accuracy'])
    return model


model_path = "cnn_lstm_bpm_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Loaded pre-trained model.")
else:
    model = build_model()
    print("Training new model...")

    class_weights = {0: 0.1, 1: 0.9}
    model.fit(X_train, y_train, epochs=30, batch_size=16, class_weight=class_weights)

    model.save(model_path)
    print("Model saved!")

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Mean Absolute Error: {mae:.2f} BPM")
