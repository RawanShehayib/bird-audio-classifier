import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


# 1. Load
audio, sample_rate = librosa.load("firsttest.wav", sr=22050)

# 2. Inspect
print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
print(f"Samples: {len(audio)}")
print(f"Sample rate: {sample_rate}")

#3. Play the audio 
sd.play(audio, sample_rate)
sd.wait() 

# 4. Visualize
plt.figure(figsize=(10, 3))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title("Bird audio waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# The array is just a list of floats
print(audio)          # [ 0.     0.0012  0.0031 -0.0028 ... ]
print(audio[0])       # first sample — probably near 0.0 (silence)
print(audio[22050])   # first sample of second 2
print(audio[-1])      # last sample

# Slice it like any list
first_second = audio[0:22050]        # samples 0 to 22049
second_half  = audio[len(audio)//2:] # everything after the midpoi

print(np.max(audio))   # loudest moment
print(np.min(audio))   # quietest moment
print(np.mean(audio))  # should be close to 0.0 for audio