from pathlib import Path

from transformers import pipeline


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_FILE = "data/audio/sample.wav"

WHISPER_MODEL = "openai/whisper-base"

SENTIMENT_MODEL = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)


# ============================================================
# WHISPER - SPEECH TO TEXT
# ============================================================

def transcribe_audio(audio_file: str) -> str:

    print("\n" + "=" * 60)
    print("STEP 1: WHISPER SPEECH-TO-TEXT")
    print("=" * 60)

    audio_path = Path(audio_file)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print(f"Audio : {audio_path}")
    print(f"Model : {WHISPER_MODEL}")

    whisper = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=-1
    )

    result = whisper(str(audio_path))

    text = result["text"].strip()

    print("\nTranscription:")
    print(text)

    return text


# ============================================================
# TEXT SENTIMENT
# ============================================================

def analyze_sentiment(text: str):

    print("\n" + "=" * 60)
    print("STEP 2: TEXT SENTIMENT ANALYSIS")
    print("=" * 60)

    print(f"Model : {SENTIMENT_MODEL}")

    sentiment_model = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=-1
    )

    result = sentiment_model(text)[0]

    sentiment = result["label"]
    confidence = result["score"]

    print("\nSentiment   :", sentiment)
    print("Confidence  :", round(confidence, 4))

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("MULTIMODAL SENTIMENT - OFFLINE BASELINE")
    print("=" * 60)

    # Step 1
    text = transcribe_audio(AUDIO_FILE)

    # Step 2
    sentiment_result = analyze_sentiment(text)

    # Final result
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print("Audio       :", AUDIO_FILE)
    print("Text        :", text)
    print(
        "Sentiment   :",
        sentiment_result["sentiment"]
    )
    print(
        "Confidence  :",
        round(sentiment_result["confidence"], 4)
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

