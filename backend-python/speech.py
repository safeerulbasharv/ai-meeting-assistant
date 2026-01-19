import whisper
import os
import sys
import time
import traceback
import numpy as np
import soundfile as sf
import librosa

# Global model cache
_model = None


def get_model():
    global _model
    if _model is not None:
        return _model

    print("\n" + "=" * 60)
    print("🤖 LOADING WHISPER MODEL (medium, CPU)")
    print("=" * 60)

    start = time.time()
    _model = whisper.load_model("medium", device="cpu")
    print(f"✅ Model loaded in {time.time() - start:.1f}s")
    print(f"⚙ Device: {_model.device}")
    return _model


def load_audio_fixed(path):
    """
    Load audio correctly as float32 mono 16kHz
    This fixes Whisper garbage output on Windows CPU.
    """

    # Read using soundfile
    audio, sr = sf.read(path)

    # Convert stereo → mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Ensure float32
    audio = audio.astype(np.float32)

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio


def transcribe_audio(path):
    print("\n🎤 TRANSCRIBING AUDIO")
    print(f"📁 File: {os.path.basename(path)}")

    try:
        if not os.path.exists(path):
            return {"text": "[File not found]", "language": "unknown"}

        model = get_model()

        print("🔄 Decoding audio correctly...")
        audio = load_audio_fixed(path)

        print(f"🎧 Audio samples: {len(audio)}")

        print("⏳ Running Whisper...")
        start = time.time()

        result = model.transcribe(
            audio,
            fp16=False,
            language=None,   # auto-detect
            task="transcribe"
        )

        elapsed = time.time() - start

        text = result["text"].strip()
        language = result.get("language", "unknown")

        print(f"✅ Done in {elapsed:.1f}s")
        print(f"🌍 Language: {language}")
        print(f"📝 Text length: {len(text)}")

        if not text:
            text = "[No speech detected]"

        return {
            "text": text,
            "language": language
        }

    except Exception as e:
        print("\n❌ TRANSCRIPTION FAILED")
        traceback.print_exc()
        return {
            "text": f"[Error: {str(e)}]",
            "language": "unknown"
        }
