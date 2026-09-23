from transformers import pipeline


MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


def analyze_sentiment(text: str):

    print("=" * 60)
    print("TEXT SENTIMENT ANALYSIS")
    print("=" * 60)

    print("Model:", MODEL_ID)
    print("Text :", text)

    sentiment_model = pipeline(
        "sentiment-analysis",
        model=MODEL_ID
    )

    result = sentiment_model(text)[0]

    label = result["label"]
    score = result["score"]

    print("\nSentiment :", label)
    print("Confidence:", round(score, 4))

    return {
        "sentiment": label,
        "confidence": score
    }


if __name__ == "__main__":

    text = "I am very happy and satisfied with the service."

    analyze_sentiment(text)