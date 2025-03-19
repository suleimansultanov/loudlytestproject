BPM Identifier

A BPM Identifier works by analyzing the periodicity of beats in an audio signal and estimating a tempo value in Beats Per Minute (BPM). However, when a song has different tempo sections, it presents a challenge because traditional BPM detection algorithms assume a relatively stable tempo throughout the track.


Description

**BPM Identifier** is a project developed to accurately estimate the Beats Per Minute (BPM) of audio signals. This is crucial in music analysis, DJ applications, and automatic music classification. The project explores three different approaches to BPM detection:

1. **Librosa-based Approach** - Utilizing the `Librosa` package for signal processing and BPM estimation.
-python librosaPackage.py
2. **Madmom-based Approach** - Using `madmom`, a specialized library for music analysis, to determine BPM. 
-python madmomPackage.py (Pre trained Model)
3. **Deep Learning Approach** - Implementing a `TensorFlow` model with CNN-LSTM architecture for BPM prediction.
-phython tensorflowBpmIdentification.py

Features

- Three different BPM detection methodologies.
- Works with various types of audio files.
- Efficient and accurate tempo estimation.
- Comparative analysis of traditional and deep-learning-based approaches.

Implementation Details

Librosa package Approach (Simple and only estimated tempo implemented)

[`Librosa`](https://librosa.org/) is a powerful Python library for audio signal processing and analysis. It provides functionalities for feature extraction, tempo estimation, and beat tracking.

Advantages:
- Lightweight and easy to use.
- Built-in tempo estimation and beat tracking functions.
- Works well for a wide range of audio signals.

Madmom package Approach(Handling multiple tempos by segments)

[`Madmom`](https://madmom.readthedocs.io/) is a music processing library specifically designed for beat tracking, onset detection, and tempo estimation. This solution is more accurate - Instead of computing a single BPM for the entire song, the algorithm divides the audio into small overlapping time windows.
Each segment is analyzed separately, and a local BPM is calculated.
The final result can be a BPM curve over time.

Advantages:
- Optimized for real-time music analysis.
- High accuracy in BPM detection compared to traditional approaches.
- Efficient and well-suited for music signal processing tasks.

Tensorflow Approach (TensorFlow CNN-LSTM Model)

This approach leverages Convolutional Neural Networks (CNNs) for feature extraction and *ong Short-Term Memory (LSTM) networks for sequential learning. The model is trained on labeled small (for presentation only) datasets of music BPM.

Advantages:
- Can learn complex patterns in audio signals.
- More adaptive to diverse music styles.
- Improves accuracy with large datasets.

## 🏗️ Project Structure
```
├── src/
│   ├── librosaPackage.py  # BPM detection using Librosa
│   ├── madmomPackage.py   # BPM detection using Madmom
│   ├── tensorflowBpmIdentification.py # BPM detection using Deep Learning (TensorFlow)
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
For librosa package python 3.13.3 was used (Need to install numpy 1.21.4)

For Madmom and Tensorflow package python 3.9.7 was used (Need to install numpy 1.23)


Future Implementations:

-Genre classifier

Develop genre classifier for genre classification

-Real Time bpm Identification

Implement real time bpm identification for on time bpm calculation


Made by Suleiman Sultanov

