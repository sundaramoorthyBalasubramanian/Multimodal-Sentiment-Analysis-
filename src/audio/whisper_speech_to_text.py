from pathlib import Path
from transformers import pipeline


MODEL_ID = "openai/whisper-base"


def transcribe_audio(audio_file: str) -> str:

    print("=" * 60)
    print("WHISPER SPEECH-TO-TEXT")
    print("=" * 60)

    audio_path = Path(audio_file)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print(f"Audio file : {audio_path}")
    print(f"Model      : {MODEL_ID}")

    print("\nLoading Whisper model...")

    whisper = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        device=-1
    )

    print("Whisper model loaded.")

    print("\nTranscribing audio...")

    result = whisper(str(audio_path))

    text = result["text"].strip()

    print("\n" + "=" * 60)
    print("TRANSCRIPTION")
    print("=" * 60)
    print(text)
    print("=" * 60)

    return text


if __name__ == "__main__":

    audio_file = "data/audio/sample.wav"

    transcribe_audio(audio_file)