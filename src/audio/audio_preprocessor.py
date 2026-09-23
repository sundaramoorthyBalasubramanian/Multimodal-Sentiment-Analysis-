
"""
Elderly Emotion AI
Audio preprocessing module

Purpose:
    1. Load an elderly person's voice recording
    2. Validate the audio file
    3. Extract basic audio information
    4. Prepare the audio for downstream processing

This is the first component of the voice-emotion pipeline.

Pipeline:
    Audio File
        ↓
    Audio Preprocessor
        ↓
    Speech-to-Text / Voice Emotion Model
"""

from pathlib import Path

import librosa


SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def validate_audio_file(audio_path: str) -> Path:
    """
    Validate that the audio file exists and has a supported format.
    """

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    return path


def load_audio(audio_path: str, sample_rate: int = 16000):
    """
    Load audio and convert it to mono at the requested sample rate.

    Parameters:
        audio_path: Path to audio file
        sample_rate: Target sampling rate

    Returns:
        audio: Audio waveform
        sample_rate: Sampling rate
    """

    path = validate_audio_file(audio_path)

    audio, sr = librosa.load(
        path,
        sr=sample_rate,
        mono=True
    )

    return audio, sr


def get_audio_information(audio_path: str) -> dict:
    """
    Extract basic information about the audio file.
    """

    audio, sr = load_audio(audio_path)

    duration = len(audio) / sr

    information = {
        "file_name": Path(audio_path).name,
        "sample_rate": sr,
        "duration_seconds": round(duration, 2),
        "num_samples": len(audio),
        "channels": 1,
    }

    return information


if __name__ == "__main__":

    # Change this to your test audio file.
    AUDIO_FILE = "data/audio/sample.wav"

    print("\nAudio preprocessing test")
    print("-" * 40)

    try:
        info = get_audio_information(AUDIO_FILE)

        for key, value in info.items():
            print(f"{key}: {value}")

        print("-" * 40)
        print("Audio validation successful.")

    except Exception as exc:
        print(f"ERROR: {exc}")

