import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- Load the audio ---
audio, sr = librosa.load("firsttest.wav", sr=22050)

# --- Compute the mel spectrogram ---
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=128,     # number of frequency bands (rows in the plot)
    fmax=8000       # max frequency to show — 8kHz covers most bird calls
)

# Convert to decibels (log scale) — makes it human-readable
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# --- Plot both side by side ---
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

# Plot 1: Waveform
librosa.display.waveshow(audio, sr=sr, ax=axes[0])
axes[0].set_title("Waveform")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")

# Plot 2: Mel spectrogram
img = librosa.display.specshow(
    mel_spec_db,
    sr=sr,
    x_axis="time",
    y_axis="mel",
    fmax=8000,
    ax=axes[1],
    cmap="magma"    # color scheme — magma is great for spectrograms
)
axes[1].set_title("Mel spectrogram")
fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

plt.tight_layout()
plt.show()