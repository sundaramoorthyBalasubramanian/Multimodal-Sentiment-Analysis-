from pathlib import Path
import json
import sys
import time


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)


# ============================================================
# IMPORT EXISTING MODULES
# ============================================================

from whisper.speech_to_text import (
    load_whisper,
    transcribe_audio,
)

from vision.facial_emotion import (
    analyze_video,
)


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_FILE = PROJECT_ROOT / "src/data" / "audio" / "sample.wav"

VIDEO_FILE = PROJECT_ROOT / "src/data" / "vision" / "sample.mp4"

OUTPUT_FILE = PROJECT_ROOT / "src/data" / "multimodal_result.json"


# Initial fusion weights
SPEECH_WEIGHT = 0.60
FACE_WEIGHT = 0.40


# ============================================================
# EMOTION NORMALIZATION
# ============================================================

def normalize_emotion(label):

    label = str(label).lower().strip()

    mapping = {
        "happiness": "happy",
        "happiness ": "happy",
        "happy": "happy",

        "sadness": "sad",
        "sad": "sad",

        "anger": "angry",
        "angry": "angry",

        "neutral": "neutral",

        "fear": "fear",

        "disgust": "disgust",

        "contempt": "contempt",

        "surprise": "surprise",
    }

    return mapping.get(
        label,
        label
    )


# ============================================================
# TEMPORARY TEXT SENTIMENT
# ============================================================

def analyze_text_sentiment(text):

    """
    Temporary text sentiment implementation.

    Later we will replace this with a proper
    Transformer sentiment model.
    """

    text = text.lower()

    positive_words = [
        "happy",
        "good",
        "great",
        "excellent",
        "fine",
        "wonderful",
        "love",
        "lovely",
        "better",
        "enjoy",
        "enjoying",
        "nice",
        "well",
        "thank",
        "thanks",
    ]

    negative_words = [
        "sad",
        "bad",
        "terrible",
        "angry",
        "anger",
        "hate",
        "pain",
        "worried",
        "worry",
        "lonely",
        "unhappy",
        "problem",
        "difficult",
        "depressed",
    ]

    positive_count = 0
    negative_count = 0

    for word in positive_words:

        if word in text:
            positive_count += 1

    for word in negative_words:

        if word in text:
            negative_count += 1

    if positive_count > negative_count:

        return {
            "label": "happy",
            "score": 0.70,
        }

    if negative_count > positive_count:

        return {
            "label": "sad",
            "score": 0.70,
        }

    return {
        "label": "neutral",
        "score": 0.50,
    }


# ============================================================
# FACE RESULT CONVERSION
# ============================================================

def get_face_scores(face_result):

    if not face_result:

        return {}

    # Your facial_emotion.py returns emotion_scores
    emotion_scores = face_result.get(
        "emotion_scores",
        {}
    )

    normalized_scores = {}

    for emotion, score in emotion_scores.items():

        emotion = normalize_emotion(
            emotion
        )

        normalized_scores[emotion] = float(
            score
        )

    return normalized_scores


# ============================================================
# FUSION
# ============================================================

def fuse_modalities(
    speech_text,
    face_result,
):

    print()
    print("=" * 60)
    print("MULTIMODAL FUSION")
    print("=" * 60)

    # --------------------------------------------------------
    # TEXT SENTIMENT
    # --------------------------------------------------------

    text_result = analyze_text_sentiment(
        speech_text
    )

    text_emotion = normalize_emotion(
        text_result["label"]
    )

    text_confidence = float(
        text_result["score"]
    )

    print()
    print("Speech/Text")
    print("-" * 60)

    print("Text       :", speech_text)
    print("Emotion    :", text_emotion)
    print(
        "Confidence :",
        f"{text_confidence:.4f}"
    )

    # --------------------------------------------------------
    # FACIAL EMOTION
    # --------------------------------------------------------

    face_scores = get_face_scores(
        face_result
    )

    if face_scores:

        face_emotion = max(
            face_scores,
            key=face_scores.get
        )

        face_confidence = face_scores[
            face_emotion
        ]

    else:

        face_emotion = "unknown"
        face_confidence = 0.0

    print()
    print("Facial Emotion")
    print("-" * 60)

    for emotion, score in sorted(
        face_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{emotion:12s}: {score:.4f}"
        )

    print()
    print("Face dominant :", face_emotion)

    # --------------------------------------------------------
    # FUSION SCORES
    # --------------------------------------------------------

    fusion_scores = {}

    # Speech contribution
    fusion_scores[text_emotion] = (
        fusion_scores.get(
            text_emotion,
            0.0
        )
        + SPEECH_WEIGHT
        * text_confidence
    )

    # Face contribution
    for emotion, score in face_scores.items():

        fusion_scores[emotion] = (
            fusion_scores.get(
                emotion,
                0.0
            )
            + FACE_WEIGHT
            * score
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if fusion_scores:

        final_emotion = max(
            fusion_scores,
            key=fusion_scores.get
        )

        final_score = fusion_scores[
            final_emotion
        ]

        total_score = sum(
            fusion_scores.values()
        )

        if total_score > 0:

            final_confidence = (
                final_score
                / total_score
            )

        else:

            final_confidence = 0.0

    else:

        final_emotion = "unknown"
        final_confidence = 0.0

    # --------------------------------------------------------
    # RESULT OBJECT
    # --------------------------------------------------------

    result = {

        "speech": {

            "text": speech_text,

            "emotion": text_emotion,

            "confidence": text_confidence,
        },

        "face": {

            "dominant_emotion":
                face_emotion,

            "confidence":
                face_confidence,

            "emotion_scores":
                face_scores,
        },

        "fusion": {

            "speech_weight":
                SPEECH_WEIGHT,

            "face_weight":
                FACE_WEIGHT,

            "emotion_scores":
                fusion_scores,

            "final_emotion":
                final_emotion,

            "confidence":
                final_confidence,
        },
    }

    # --------------------------------------------------------
    # PRINT FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL MULTIMODAL RESULT")
    print("=" * 60)

    print()
    print(
        "Speech emotion :",
        text_emotion
    )

    print(
        "Face emotion   :",
        face_emotion
    )

    print()
    print("Fusion scores")
    print("-" * 60)

    for emotion, score in sorted(
        fusion_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{emotion:12s}: {score:.4f}"
        )

    print()
    print(
        "FINAL EMOTION  :",
        final_emotion
    )

    print(
        "CONFIDENCE     :",
        f"{final_confidence:.4f}"
    )

    print("=" * 60)

    return result


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    total_start = time.time()

    print("=" * 60)
    print("MULTIMODAL SENTIMENT ANALYSIS")
    print("=" * 60)

    print()
    print("Audio :", AUDIO_FILE)
    print("Video :", VIDEO_FILE)

    # --------------------------------------------------------
    # VALIDATE FILES
    # --------------------------------------------------------

    if not AUDIO_FILE.exists():

        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    if not VIDEO_FILE.exists():

        raise FileNotFoundError(
            f"Video file not found: {VIDEO_FILE}"
        )

    # ========================================================
    # STEP 1 — WHISPER
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 1 — WHISPER SPEECH-TO-TEXT")
    print("=" * 60)

    whisper = load_whisper()

    speech_text = transcribe_audio(
        whisper,
        AUDIO_FILE
    )

    # ========================================================
    # STEP 2 — FACIAL EMOTION
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 2 — FACIAL EMOTION")
    print("=" * 60)

    face_result = analyze_video(
        str(VIDEO_FILE)
    )

    if face_result is None:

        print(
            "No facial emotion result available."
        )

        face_result = {
            "emotion_scores": {}
        }

    # ========================================================
    # STEP 3 — FUSION
    # ========================================================

    print()
    print("=" * 60)
    print("STEP 3 — MULTIMODAL FUSION")
    print("=" * 60)

    final_result = fuse_modalities(
        speech_text,
        face_result
    )

    # ========================================================
    # STEP 4 — SAVE RESULT
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_result,
            f,
            indent=4
        )

    total_time = (
        time.time()
        - total_start
    )

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(
        "Final emotion :",
        final_result["fusion"]["final_emotion"]
    )

    print(
        "Confidence    :",
        f"{final_result['fusion']['confidence']:.4f}"
    )

    print(
        "Total time    :",
        f"{total_time:.2f} seconds"
    )

    print()
    print(
        "JSON result   :",
        OUTPUT_FILE
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

