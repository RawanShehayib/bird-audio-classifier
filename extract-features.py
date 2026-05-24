import os
import librosa
import numpy as np

'''
The script opens every audio file, chops it into 3-second pieces, 
converts each piece into 16,640 numbers using a mel spectrogram, 
pairs each set of numbers with the species name, and saves the whole table to disk
'''

# --- Config ---
DATASET_PATH = r"C:\Users\sheha\OneDrive\Desktop\DS-final-project\dataset"
SPECIES = ["robin", "chaffinch", "great_tit"]
TARGET_SR = 22050

'''chop audio into 3-second pieces, 
use 128 frequency bands in the spectrogram, 
and only look at frequencies up to 8000 Hz'''

WINDOW_SECONDS = 3       
N_MELS = 128                
FMAX = 8000
MAX_CLIPS = 200  # cap every species at 200 clips for a balanced dataset

features = []   #  a flattened mel spectrogram (array of numbers)
labels = []     #  the species name (string)

window_samples = WINDOW_SECONDS * TARGET_SR  # = 66,050 samples per window

print("=== Extracting features ===\n")

'''This is two nested loops. The outer loop goes through each species folder (robin, chaffinch, great_tit).
 The inner loop goes through every file inside that folder. For each file, it loads the audio into a numpy array '''

for species in SPECIES:
    folder = os.path.join(DATASET_PATH, species)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".mp3"))]
    
    clip_count = 0

    for fname in files:
        if clip_count >= MAX_CLIPS:  # stop opening new files once cap is reached
            break

        path = os.path.join(folder, fname)
        
        try:
            audio, sr = librosa.load(path, sr=TARGET_SR)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            continue

        '''explanation on the 3-second window process:
         grabbing what's inside the 3-second frame, processing it, then moving forward 3 seconds and repeating. 
         Each step is one "window" — one training clip. In this way, a 3-minute robin recording becomes 60 clips, 
         which creates more data
        '''
        start = 0
        while start + window_samples <= len(audio):
            if clip_count >= MAX_CLIPS:  # stop mid-file if cap is reached
                break

            window = audio[start : start + window_samples]

            # Compute mel spectrogram for this window
           # mel = librosa.feature.melspectrogram(
            #    y=window,
             #   sr=TARGET_SR,
              ## fmax=FMAX
            #)
            # Replace your current mel spectrogram block with this
            mel = librosa.feature.melspectrogram(
            y=window, sr=TARGET_SR, n_mels=N_MELS, fmax=FMAX)
            mel_db = librosa.power_to_db(mel, ref=np.max)

# MFCCs capture tonal quality — very good at distinguishing
# chaffinch's descending flourish from great tit's two-note call
            mfcc = librosa.feature.mfcc(
            y=window, sr=TARGET_SR, n_mfcc=40)

# Spectral centroid — captures where the sound energy is centered
            centroid = librosa.feature.spectral_centroid(
            y=window, sr=TARGET_SR)

# Combine everything into one feature vector
            feature_vector = np.concatenate([
            mel_db.flatten(),      # 16,640 numbers
            mfcc.flatten(),        # 5,200 numbers
            centroid.flatten()     #   130 numbers
            ])
# Total: 21,840 features per clip
            '''
            First (mel) melspectrogram turns the 3-second audio window into a 2D grid (a small image of the sound)
            power_to_db (mel_db) converts the values to decibels so the contrast is visible.
            Then, flatten (feature_vector) unrolls that 2D grid into one long 1D list of numbers (one row in the table)
            '''
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Flatten 2D spectrogram → 1D feature vector
            feature_vector = mel_db.flatten()

            # Store in parallel lists
            features.append(feature_vector)
            labels.append(species)

            clip_count += 1
            start += window_samples  # move to next window (no overlap for now)

    print(f"{species}: {len(files)} files → {clip_count} clips extracted")

# Convert lists to numpy arrays (required for scikit-learn)
features = np.array(features)
labels   = np.array(labels)

print(f"\nfeatures shape: {features.shape}")
print(f"labels shape:   {labels.shape}")
print(f"Feature vector size per clip: {features.shape[1]}")
print(f"Label examples: {labels[:5]}")
print(f"\nAll unique labels: {np.unique(labels)}")

# Save to disk so you don't have to recompute every time
np.save("features.npy", features)
np.save("labels.npy",   labels)
print("\nSaved features.npy and labels.npy")