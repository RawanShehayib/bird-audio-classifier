import matplotlib
matplotlib.use('Agg')  # must be before any other matplotlib import
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import joblib
import tempfile
import os
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

model  = joblib.load("bird_model.pkl")
scaler = joblib.load("scaler.pkl")

TARGET_SR      = 22050
WINDOW_SECONDS = 3
N_MELS         = 128
FMAX           = 8000

SPECIES_INFO = {
    "robin":     "European Robin",
    "chaffinch": "Common Chaffinch",
    "great_tit": "Great Tit"
}

VOTE_THRESHOLD = 40.0

def spectrogram_to_base64(audio_segment, sr):
    '''
    Same as testingmelspec.py but instead of plt.show(),
    converts the image to base64 so React can display it
    '''
    mel_spec = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=sr,
        n_mels=N_MELS,
        fmax=FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 2))
    img = librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmax=FMAX,
        ax=ax,
        cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # ← add these lines to make everything white
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")

    # make colorbar white too
    fig.axes[-1].tick_params(colors="white")
    fig.axes[-1].yaxis.label.set_color("white")

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#161b22')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Load audio
        audio, sr = librosa.load(tmp_path, sr=TARGET_SR)
        duration  = round(len(audio) / sr, 1)

        # Sliding window feature extraction
        window_samples  = WINDOW_SECONDS * TARGET_SR
        feature_vectors = []
        audio_windows   = []  # store raw audio per window for spectrograms
        start = 0

        while start + window_samples <= len(audio):
            window = audio[start : start + window_samples]
            audio_windows.append(window)  # save raw audio

            mel    = librosa.feature.melspectrogram(
                        y=window, sr=TARGET_SR, n_mels=N_MELS, fmax=FMAX)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            feature_vectors.append(mel_db.flatten())
            start += window_samples

        if len(feature_vectors) == 0:
            return jsonify({"error": "Recording too short — needs at least 3 seconds"}), 400

        # Normalize and predict
        feature_vectors = np.array(feature_vectors)
        feature_vectors = scaler.transform(feature_vectors)
        predictions     = model.predict(feature_vectors)
        confidences     = model.predict_proba(feature_vectors)

        # Majority vote
        species, counts = np.unique(predictions, return_counts=True)
        vote_map        = dict(zip(species.tolist(), counts.tolist()))
        winner          = max(vote_map, key=vote_map.get)

        # Build confidence map
        avg_proba      = confidences.mean(axis=0)
        class_names    = model.classes_.tolist()
        confidence_map = {
            cls: round(float(avg_proba[i]) * 100, 1)
            for i, cls in enumerate(class_names)
        }

        # Build timeline
        timeline = [
            {
                "start":      i * WINDOW_SECONDS,
                "end":        (i + 1) * WINDOW_SECONDS,
                "prediction": str(predictions[i]),
                "confidence": round(float(max(confidences[i])) * 100, 1)
            }
            for i in range(len(predictions))
        ]

        # Generate spectrograms
        full_spec_b64 = spectrogram_to_base64(audio, sr)
        window_specs  = [
            spectrogram_to_base64(w, sr)
            for w in audio_windows
        ]

        # Reject if winner didn't get enough window votes
        total_windows   = len(predictions)
        winner_votes    = vote_map.get(winner, 0)
        winner_vote_pct = (winner_votes / total_windows) * 100

        if winner_vote_pct < VOTE_THRESHOLD:
            return jsonify({
                "prediction":  "unknown",
                "common_name": "No match found",
                "duration":    duration,
                "windows":     total_windows,
                "votes":       vote_map,
                "confidence":  confidence_map,
                "rejected":    True,
                "timeline":    timeline,
                "spectrogram": full_spec_b64,
                "window_specs": window_specs
            })

        return jsonify({
            "prediction":   winner,
            "common_name":  SPECIES_INFO.get(winner, winner),
            "duration":     duration,
            "windows":      total_windows,
            "votes":        vote_map,
            "confidence":   confidence_map,
            "timeline":     timeline,
            "spectrogram":  full_spec_b64,
            "window_specs": window_specs
        })

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)