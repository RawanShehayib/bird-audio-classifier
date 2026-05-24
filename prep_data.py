import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

'''
this program aims to make the data ready for traning the model.
it normalizes the raw spectrogram values and splits the data between training and the actual test
The model doesn't know how to open wav files or read folders — it only speaks numpy arrays. prep_data.py was the translator.
Scikit-learn needs this translated data to be able to work 
(X_train  shape: (516, 16640)  ← 2D array, floats, normalized
y_train  shape: (516,)        ← 1D array, species name strings
X_test   shape: (130, 16640)  ← 2D array, floats, normalized
y_test   shape: (130,)        ← 1D array, species name strings)
'''

# Load what extract_features.py saved
features = np.load("features.npy")
labels   = np.load("labels.npy")

print(f"  Total clips:  {len(features)}")
print(f"  Feature size: {features.shape[1]}")

''' Hash map — count clips per species
analogy: opening a box of 646 flashcards and sorting them into three piles by species to see if the piles are roughly equal.
'''
label_counts = Counter(labels)
print(f"  Clips per species: {dict(label_counts)}")

''' Normalizing: 
rescaling every feature to the same range
analogy: imagine grading students where one exam was out of 10 points and another out of 1000 points. 
'''
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print(f"  Mean before: {features.mean():.1f}  →  after: {features_scaled.mean():.1f}")

'''Train/test split
Training set (80%) → 516 clips → model LEARNS from these
Test set     (20%) → 130 clips → model NEVER sees these until exam time
'''
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels,
    test_size=0.2,
    random_state=42, #The randomness was decided once by the seed — after that it's just following a fixed recipe.
    stratify=labels # makes sure both piles have the same ratio of species
)
print(f"\n=== Train/test split ===")
print(f"  Training clips: {len(X_train)}")
print(f"  Test clips:     {len(X_test)}")

'''Save everything
X_train.npy  → the 516 training feature vectors
X_test.npy   → the 130 test feature vectors  
y_train.npy  → the 516 training labels
y_test.npy   → the 130 test labels
scaler.pkl   → the normalization settings
'''

np.save("X_train.npy", X_train)
np.save("X_test.npy",  X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy",  y_test)

joblib.dump(scaler, "scaler.pkl")

print("  X_train.npy, X_test.npy, y_train.npy, y_test.npy, scaler.pkl")
