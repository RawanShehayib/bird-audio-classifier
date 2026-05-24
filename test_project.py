import unittest
import numpy as np
import librosa
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from collections import Counter, deque
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestFeatureExtraction(unittest.TestCase):
    '''Test 2 — Mel spectrogram feature extraction'''

    def setUp(self):
        self.TARGET_SR      = 22050
        self.WINDOW_SECONDS = 3
        self.N_MELS         = 128
        self.FMAX           = 8000
        self.window_samples = self.WINDOW_SECONDS * self.TARGET_SR
        # create a synthetic 3-second audio clip for testing
        self.test_audio     = np.random.uniform(-0.1, 0.1, self.window_samples)

    def test_mel_spectrogram_shape(self):
        '''Mel spectrogram has correct shape (128 rows)'''
        mel = librosa.feature.melspectrogram(
            y=self.test_audio,
            sr=self.TARGET_SR,
            n_mels=self.N_MELS,
            fmax=self.FMAX
        )
        self.assertEqual(mel.shape[0], 128)  # 128 frequency bands
        self.assertGreater(mel.shape[1], 0)  # at least 1 time step
        print(f"  PASS: spectrogram shape = {mel.shape}")

    def test_feature_vector_size(self):
        '''Flattened feature vector has 16,640 numbers'''
        mel = librosa.feature.melspectrogram(
            y=self.test_audio,
            sr=self.TARGET_SR,
            n_mels=self.N_MELS,
            fmax=self.FMAX
        )
        mel_db         = librosa.power_to_db(mel, ref=np.max)
        feature_vector = mel_db.flatten()
        self.assertEqual(len(feature_vector), 128 * mel.shape[1])
        print(f"  PASS: feature vector size = {len(feature_vector)}")

    def test_decibel_conversion(self):
        '''power_to_db produces values in valid dB range'''
        mel    = librosa.feature.melspectrogram(
            y=self.test_audio, sr=self.TARGET_SR,
            n_mels=self.N_MELS, fmax=self.FMAX)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        self.assertLessEqual(np.max(mel_db), 0.1)   # max should be ~0 dB
        self.assertGreater(np.min(mel_db), -100)     # min should be above -100 dB
        print(f"  PASS: dB range = [{np.min(mel_db):.1f}, {np.max(mel_db):.1f}] dB")

class TestAudioLoading(unittest.TestCase):
    '''Test 1 — Audio loading produces correct array structure'''

    def setUp(self):
        # find any wav file in dataset to test with
        self.test_file = None
        dataset_path   = r"C:\Users\sheha\OneDrive\Desktop\DS-final-project\dataset"
        for species in ["robin", "chaffinch", "great_tit"]:
            folder = os.path.join(dataset_path, species)
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder)
                         if f.lower().endswith((".wav", ".mp3"))]
                if files:
                    self.test_file = os.path.join(folder, files[0])
                    break

    def test_audio_loads_as_array(self):
        '''Audio file loads as a 1D numpy array'''
        if self.test_file is None:
            self.skipTest("No audio files found in dataset")
        audio, sr = librosa.load(self.test_file, sr=22050)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.ndim, 1)  # must be 1D
        print(f"  PASS: audio loaded as 1D array, {len(audio)} samples")

    def test_sample_rate_is_correct(self):
        '''Sample rate is set to 22,050'''
        if self.test_file is None:
            self.skipTest("No audio files found in dataset")
        audio, sr = librosa.load(self.test_file, sr=22050)
        self.assertEqual(sr, 22050)
        print(f"  PASS: sample rate = {sr}")

    def test_audio_values_in_range(self):
        '''Audio values are between -1.0 and +1.0'''
        if self.test_file is None:
            self.skipTest("No audio files found in dataset")
        audio, sr = librosa.load(self.test_file, sr=22050)
        self.assertLessEqual(np.max(audio), 1.0)
        self.assertGreaterEqual(np.min(audio), -1.0)
        print(f"  PASS: audio values in range [{np.min(audio):.3f}, {np.max(audio):.3f}]")
if __name__ == "__main__":
    print("=" * 60)
    print("Bird Audio Classifier — Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)