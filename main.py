from typing import Dict, Any, Optional
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import uvicorn
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import whisper   # <--- Whisper for multilingual speech recognition

# Load Whisper model (tiny or small for speed, medium/large for accuracy)
whisper_model = whisper.load_model("small")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = GoogleTranslator(source='auto', target='en')


class AnalysisResponse(BaseModel):
    status: str
    thresholds: Dict[str, float]
    message: Optional[str] = None


@app.get("/health")
async def check():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        raw_path = f"temp_input_{audio.filename}"
        with open(raw_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)

        # Convert any format to WAV
        audio_path = "temp_audio.wav"
        try:
            sound = AudioSegment.from_file(raw_path)
            sound.export(audio_path, format="wav")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error converting file: {str(e)}")

        # Extract features
        features = extract_features(audio_path)

        # Risk score
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
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")


def extract_features(audio_path: str):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # --- Faster load: only first 30 seconds ---
    y, sample_rate = librosa.load(audio_path, sr=16000, duration=30.0)
    duration = librosa.get_duration(y=y, sr=sample_rate)

    # --- Whisper Transcription (multilingual) ---
    result = whisper_model.transcribe(audio_path, language=None)  # auto-detect
    text = result.get("text", "").strip()
    detected_lang = result.get("language", "en")

    # --- Translate to English if not English ---
    translated_text = text
    if detected_lang != "en" and text:
        try:
            translated_text = translator.translate(text)
        except Exception:
            translated_text = text

    # --- Features ---
    frame_length = 2048
    hop_length = 512
    energy = np.array([
        float(np.sum(np.abs(y[int(i):int(i+frame_length)])**2))
        for i in range(0, len(y), hop_length)
    ], dtype=float)
    threshold = np.mean(energy) * 0.1
    pauses = energy < threshold
    pause_count = np.sum(pauses)
    total_pause_duration = (pause_count * hop_length) / sample_rate

    words = word_tokenize(translated_text) if translated_text else []
    words_per_minute = (len(words) / duration) * 60 if duration > 0 else 0
    lexical_diversity = (len(set(words)) / len(words)) if len(words) > 0 else 0
    fillers = ['um', 'uh', 'ah', 'like', 'you know']
    filler_count = sum(1 for word in words if word.lower() in fillers)

    return {
        'pause_count': int(pause_count),
        'total_pause_duration': round(total_pause_duration, 2),
        'words_per_minute': round(words_per_minute, 2),
        'lexical_diversity': round(lexical_diversity, 2),
        'filler_count': int(filler_count),
        'duration': round(duration, 2),
        'transcript_original': text,
        'transcript_english': translated_text,
        'detected_language': detected_lang
    }


def calculate_risk(features: Dict[str, Any]) -> float:
    score = 0.0
    thresholds = {
        'pause_count': 393.0,
        'words_per_minute': 36.2975,
        'lexical_diversity': 0.82,
        'filler_count': 0.0
    }

    if features['pause_count'] > thresholds['pause_count']:
        score += min(0.4, (features['pause_count'] - thresholds['pause_count']) / 100)
    if features['words_per_minute'] < thresholds['words_per_minute']:
        score += min(0.3, (thresholds['words_per_minute'] - features['words_per_minute']) / 50)
    if features['lexical_diversity'] < thresholds['lexical_diversity']:
        score += min(0.2, (thresholds['lexical_diversity'] - features['lexical_diversity']) * 2)
    if features['filler_count'] > thresholds['filler_count']:
        score += min(0.1, (features['filler_count'] - thresholds['filler_count']) / 10)

    return min(score, 1.0)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
