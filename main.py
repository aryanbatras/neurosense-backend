from typing import Dict, Any, List, Optional, Tuple
import os
import re
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa
import speech_recognition as sr
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import uvicorn
from langdetect import detect_langs, DetectorFactory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Initialize langdetect
DetectorFactory.seed = 0

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Small dictionaries of filler tokens
_EN_FILLERS = {'um', 'uh', 'ah', 'hmm', 'mm', 'like', 'you know', 'i mean', 'sort of', 'kind of', 'well'}
_HI_FILLERS = {'um', 'uh', 'achha', 'accha', 'haan', 'nahi', 'nahin', 'bas', 'toh', 'to', 'dekho', 'matlab', 'yaani',
               'yaar'}
_HI_COMMON = {'hai', 'kya', 'nahi', 'haan', 'maine', 'mujhe', 'mera', 'meri', 'karna', 'karta', 'tha', 'thi', 'kaun',
              'kyun', 'kuch', 'acha', 'accha', 'raha', 'rahi', 'bola', 'bol'}


class AnalysisResponse(BaseModel):
    status: str
    thresholds: Dict[str, float]
    message: Optional[str] = None


def transcribe_audio(audio_path: str) -> Tuple[str, str]:
    """
    Returns (original_transcript, detected_language_code)
    Uses speech_recognition Google (tries en-IN then hi-IN)
    """
    import speech_recognition as sr_mod
    r = sr_mod.Recognizer()
    text_en = ""
    text_hi = ""
    try:
        with sr_mod.AudioFile(audio_path) as source:
            audio = r.record(source)
        # Try english first
        try:
            text_en = r.recognize_google(audio, language="en-IN")
        except Exception:
            text_en = ""
        # Try hindi as fallback
        try:
            text_hi = r.recognize_google(audio, language="hi-IN")
        except Exception:
            text_hi = ""

        # Decide which to use by heuristics: prefer non-empty, or longer text
        chosen = text_en if len(text_en) >= len(text_hi) else text_hi
        lang = detect_language_from_text(chosen or text_en or text_hi)
        return (chosen.strip(), lang)
    except Exception as e:
        print("speech_recognition ASR failed:", e)
        return ("", "unknown")


def detect_language_from_text(text: str) -> str:
    """
    Heuristics to decide language:
    1) If Devanagari characters present -> 'hi'
    2) Else use langdetect.detect_langs for probabilities (needs some text)
    3) Else fallback by searching common Hindi words (helps with romanized Hindi)
    """
    if not text or len(text.strip()) == 0:
        return "unknown"
    # 1) Devanagari script check
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"
    # 2) try langdetect
    try:
        langs = detect_langs(text)
        if len(langs) > 0:
            top = langs[0]
            code = top.lang
            prob = getattr(top, 'prob', 0.0)
            if code == 'hi' and prob >= 0.60:
                return 'hi'
            if code == 'en' and prob >= 0.60:
                return 'en'
    except Exception:
        pass
    # 3) romanized Hindi heuristic: presence of common Hindi words
    tw = set(w.lower() for w in re.findall(r'\w+', text))
    if len(tw & _HI_COMMON) >= 1:
        return 'hi'
    # default to English if uncertain
    return 'en'


def extract_features(audio_path: str) -> Dict[str, Any]:
    """
    Returns dict with:
      - language: 'en' / 'hi' / 'unknown'
      - original_transcript (native)
      - duration, pause_count, total_pause_duration, avg_pause_length, pause_rate
      - words_per_minute, lexical_diversity, filler_count
    """
    # --- audio load & pause/voiced detection ---
    y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=60.0)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # voiced intervals with librosa
    voiced_intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
    voiced_times = [(s / sr, e / sr) for s, e in voiced_intervals]

    pause_count = 0
    total_pause_duration = 0.0
    if len(voiced_times) == 0:
        pause_count = 1
        total_pause_duration = duration
    else:
        if voiced_times[0][0] > 0.05:
            pause_count += 1
            total_pause_duration += voiced_times[0][0]
        for i in range(len(voiced_times) - 1):
            gap = voiced_times[i + 1][0] - voiced_times[i][1]
            if gap > 0.10:
                pause_count += 1
                total_pause_duration += gap
        last_end = voiced_times[-1][1]
        if duration - last_end > 0.05:
            pause_count += 1
            total_pause_duration += (duration - last_end)

    avg_pause_length = (total_pause_duration / pause_count) if pause_count > 0 else 0.0
    pause_rate = total_pause_duration / max(duration, 1e-6)

    # --- ASR & language detection ---
    original_transcript, detected_lang = transcribe_audio(audio_path)

    # --- textual features ---
    text_for_metrics = original_transcript
    words = word_tokenize(text_for_metrics) if text_for_metrics else []
    word_count = len(words)
    words_per_minute = (word_count / duration) * 60 if duration > 0 else 0.0
    lexical_diversity = (len(set([w.lower() for w in words])) / word_count) if word_count > 0 else 0.0

    # Count fillers
    filler_count = 0
    orig_tokens = [w.lower() for w in re.findall(r'\w+', original_transcript)] if original_transcript else []
    filler_count += sum(1 for w in orig_tokens if w in _HI_FILLERS) if detected_lang == 'hi' else 0
    filler_count += sum(1 for w in orig_tokens if w in _EN_FILLERS)

    return {
        'language': detected_lang,
        'original_transcript': original_transcript,
        'duration': round(duration, 2),
        'pause_count': int(pause_count),
        'total_pause_duration': round(total_pause_duration, 3),
        'avg_pause_length': round(avg_pause_length, 3),
        'pause_rate': round(pause_rate, 3),
        'word_count': int(word_count),
        'words_per_minute': round(words_per_minute, 2),
        'lexical_diversity': round(lexical_diversity, 3),
        'filler_count': int(filler_count)
    }


def calculate_risk(features: Dict[str, Any]) -> float:
    """Calculate dementia risk score using data-driven thresholds"""
    score: float = 0.0

    # Updated thresholds from analysis
    thresholds = {
        'pause_count': 393.0,
        'words_per_minute': 36.2975,
        'lexical_diversity': 0.82,
        'filler_count': 0.0
    }

    # Weight features based on their importance in dementia detection
    if features['pause_count'] > thresholds['pause_count']:
        score += min(0.4, (features['pause_count'] - thresholds['pause_count']) / 100)

    if features['words_per_minute'] < thresholds['words_per_minute']:
        score += min(0.3, (thresholds['words_per_minute'] - features['words_per_minute']) / 10)

    if features['lexical_diversity'] < thresholds['lexical_diversity']:
        score += min(0.2, (thresholds['lexical_diversity'] - features['lexical_diversity']) * 2)

    if features['filler_count'] > thresholds['filler_count']:
        score += min(0.1, (features['filler_count'] - thresholds['filler_count']) / 1)

    return min(score, 1.0)


@app.get("/health")
async def check():
    try:
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    if not audio.filename.lower().endswith(('.wav', '.mp3', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a WAV, MP3, or OGG file.")

    try:
        # Save the uploaded file temporarily
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)

        features = extract_features(audio_path)
        risk_score = calculate_risk(features)

        if risk_score < 0.3:
            message = "Low risk detected. No significant indicators found."
        elif risk_score < 0.7:
            message = "Moderate risk detected. Consider clinical evaluation."
        else:
            message = "High risk detected. Professional evaluation recommended."

        return {
            "risk_score": round(risk_score, 2),
            "message": message,
            "features": features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_csv")
async def generate_csv(dataset_path: str = "audio", output_csv: str = "features.csv"):
    """
    Walk through dataset/audio folders, extract features, label dementia=1 / control=0,
    and save into a CSV.
    """
    import pandas as pd
    rows = []

    dementia_path = os.path.join(dataset_path, "Dementia")
    control_path = os.path.join(dataset_path, "Control")

    counter = 0
    # --- Dementia files ---
    for root, _, files in os.walk(dementia_path):
        for file in files:
            counter += 1
            print(f"Processing {counter}: {file}")
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    feats = extract_features(file_path)
                    feats["label"] = 1
                    feats["filename"] = file
                    feats["person_id"] = os.path.basename(root)  # the subfolder name
                    rows.append(feats)
                except Exception as e:
                    print(f"Error processing Dementia/{file}: {e}")

    # --- Control files ---
    for root, _, files in os.walk(control_path):
        for file in files:
            counter += 1
            print(f"Processing {counter}: {file}")
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    feats = extract_features(file_path)
                    feats["label"] = 0
                    feats["filename"] = file
                    feats["person_id"] = os.path.basename(root)
                    rows.append(feats)
                except Exception as e:
                    print(f"Error processing Control/{file}: {e}")

    if not rows:
        raise HTTPException(status_code=500, detail="No audio files processed!")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    return {
        "status": "success",
        "rows": len(rows),
        "output_csv": output_csv
    }

@app.post("/train_model")
async def train_model(csv_file: UploadFile = File(...)):
    """
    Upload features.csv and train ML model.
    Saves dementia_model.pkl and returns test metrics.
    """
    try:
        contents = await csv_file.read()
        csv_path = "uploaded_features.csv"
        with open(csv_path, "wb") as f:
            f.write(contents)

        df = pd.read_csv(csv_path)

        required = {"pause_count", "words_per_minute", "lexical_diversity", "filler_count", "label"}
        if not required.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"CSV missing required columns {required}")

        X = df[["pause_count", "words_per_minute", "lexical_diversity", "filler_count"]]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        report = classification_report(y_test, model.predict(X_test), output_dict=True)

        joblib.dump(model, "dementia_model.pkl")

        return {
            "status": "success",
            "model_file": "dementia_model.pkl",
            "metrics": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)