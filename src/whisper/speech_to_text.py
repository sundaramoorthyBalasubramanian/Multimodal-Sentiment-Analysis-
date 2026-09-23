from pathlib import Path
import time

from transformers import pipeline


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "openai/whisper-base"

AUDIO_FILE = Path("data/audio/sample.wav")


# ============================================================
# LOAD WHISPER
# ============================================================

def load_whisper():

    print("=" * 60)
    print("Loading Whisper model")
    print("=" * 60)

    print(f"Model: {MODEL_ID}")

    start_time = time.time()

    whisper = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_ID,
        device=-1,              # CPU
    )

    elapsed = time.time() - start_time

    print(f"Whisper loaded in {elapsed:.2f} seconds")

    return whisper


# ============================================================
# SPEECH TO TEXT
# ============================================================

def transcribe_audio(whisper, audio_file):

    print()
    print("=" * 60)
    print("Speech-to-Text")
    print("=" * 60)

    print(f"Audio: {audio_file}")

    if not audio_file.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_file}"
        )

    start_time = time.time()

    result = whisper(
        str(audio_file)
    )

    elapsed = time.time() - start_time

    text = result["text"].strip()

    print()
    print("Transcription:")
    print("-" * 60)
    print(text)
    print("-" * 60)

    print(f"Transcription time: {elapsed:.2f} seconds")

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("OFFLINE WHISPER SPEECH-TO-TEXT")
    print("=" * 60)

    whisper = load_whisper()

    text = transcribe_audio(
        whisper,
        AUDIO_FILE
    )

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(text)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()