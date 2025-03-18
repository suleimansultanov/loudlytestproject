BPM Identifier

![GitHub repo size](https://img.shields.io/github/repo-size/your-username/your-repo-name)
![GitHub stars](https://img.shields.io/github/stars/your-username/your-repo-name?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/your-repo-name?style=social)
![GitHub license](https://img.shields.io/github/license/your-username/your-repo-name)

Description

**BPM Identifier** is a project developed to accurately estimate the Beats Per Minute (BPM) of audio signals. This is crucial in music analysis, DJ applications, and automatic music classification. The project explores three different approaches to BPM detection:

1. **Librosa-based Approach** - Utilizing the `Librosa` package for signal processing and BPM estimation.
2. **Madmom-based Approach** - Using `madmom`, a specialized library for music analysis, to determine BPM.
3. **Deep Learning Approach** - Implementing a `TensorFlow` model with CNN-LSTM architecture for BPM prediction.

Features

- Three different BPM detection methodologies.
- Works with various types of audio files.
- Efficient and accurate tempo estimation.
- Comparative analysis of traditional and deep-learning-based approaches.

Implementation Details

Librosa package Approach

[`Librosa`](https://librosa.org/) is a powerful Python library for audio signal processing and analysis. It provides functionalities for feature extraction, tempo estimation, and beat tracking.

Advantages:
- Lightweight and easy to use.
- Built-in tempo estimation and beat tracking functions.
- Works well for a wide range of audio signals.

Madmom package Approach

[`Madmom`](https://madmom.readthedocs.io/) is a music processing library specifically designed for beat tracking, onset detection, and tempo estimation.

Advantages:
- Optimized for real-time music analysis.
- High accuracy in BPM detection compared to traditional approaches.
- Efficient and well-suited for music signal processing tasks.

Tensorflow Approach (TensorFlow CNN-LSTM Model)

This approach leverages Convolutional Neural Networks (CNNs) for feature extraction and *ong Short-Term Memory (LSTM) networks for sequential learning. The model is trained on labeled small (for presenatation only) datasets of music BPM.

Advantages:
- Can learn complex patterns in audio signals.
- More adaptive to diverse music styles.
- Improves accuracy with large datasets.

## 🏗️ Project Structure
```
├── src/
│   ├── librosaPackage.py  # BPM detection using Librosa
│   ├── madmomPackage.py   # BPM detection using Madmom
│   ├── tensorflowCnnLstm.py # BPM detection using Deep Learning (TensorFlow)
│   └── ...
├── README.md  # Documentation
├── requirements.txt  # Dependencies
└── ...
```
## 🖥️ Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run BPM Detection

- **Librosa Approach:**
  ```bash
  python src/bpm_librosa.py --input your_audio_file.wav
  ```
- **Madmom Approach:**
  ```bash
  python src/bpm_madmom.py --input your_audio_file.wav
  ```
- **TensorFlow CNN-LSTM Approach:**
  ```bash
  python src/bpm_cnn_lstm.py --input your_audio_file.wav
  ```

Made by Suleiman Sultanov

