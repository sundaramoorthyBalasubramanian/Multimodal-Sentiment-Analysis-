
"""
Generate a sample WAV voice recording for testing
the elderly emotion AI pipeline.
"""

import pyttsx3
from pathlib import Path


OUTPUT_FILE = Path("data/audio/sample.wav")

TEXT = (
    "I am feeling happy today. "
    "My daughter called me this morning and "
    "I enjoyed talking to her."
)


def generate_audio():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    engine = pyttsx3.init()

    # Speech rate
    engine.setProperty("rate", 140)

    # Volume
    engine.setProperty("volume", 1.0)

    print("Generating audio...")
    print(f"Text: {TEXT}")

    engine.save_to_file(
        TEXT,
        str(OUTPUT_FILE)
    )

    engine.runAndWait()

    print(f"\nAudio created: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_audio()

