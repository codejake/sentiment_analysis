import json
from typing import Dict
from transformers import pipeline
from rich import print as rprint


class SentimentAnalyzer:
    def __init__(self):
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
        )

        self.sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True,
        )

    def analyze_headline(self, headline: str) -> Dict[str, float]:
        if not headline or not isinstance(headline, str):
            raise ValueError("Headline must be a non-empty string")

        headline = headline.strip()
        if not headline:
            raise ValueError("Headline cannot be empty or whitespace only")

        emotion_results = self.emotion_classifier(headline)[0]
        sentiment_results = self.sentiment_classifier(headline)[0]

        emotion_scores = {
            result["label"].lower(): result["score"] for result in emotion_results
        }
        sentiment_scores = {
            result["label"].lower(): result["score"] for result in sentiment_results
        }

        scores = {
            "optimism": self._calculate_optimism(emotion_scores, sentiment_scores),
            "pessimism": self._calculate_pessimism(emotion_scores, sentiment_scores),
            "anger": emotion_scores.get("anger", 0.0),
            "hopeful": self._calculate_hopeful(emotion_scores, sentiment_scores),
            "sad": emotion_scores.get("sadness", 0.0),
            "fear": emotion_scores.get("fear", 0.0),
            "uncertainty": self._calculate_uncertainty(headline, emotion_scores),
            "anxiety": self._calculate_anxiety(emotion_scores),
            "confidence": self._calculate_confidence(emotion_scores, sentiment_scores),
            "urgency": self._calculate_urgency(headline, emotion_scores),
        }

        return {k: round(v, 4) for k, v in scores.items()}

    def _calculate_optimism(
        self, emotions: Dict[str, float], sentiment: Dict[str, float]
    ) -> float:
        joy = emotions.get("joy", 0.0)
        positive = sentiment.get("positive", sentiment.get("label_2", 0.0))
        return min(1.0, (joy * 0.7 + positive * 0.3))

    def _calculate_pessimism(
        self, emotions: Dict[str, float], sentiment: Dict[str, float]
    ) -> float:
        sadness = emotions.get("sadness", 0.0)
        negative = sentiment.get("negative", sentiment.get("label_0", 0.0))
        return min(1.0, (sadness * 0.6 + negative * 0.4))

    def _calculate_hopeful(
        self, emotions: Dict[str, float], sentiment: Dict[str, float]
    ) -> float:
        joy = emotions.get("joy", 0.0)
        positive = sentiment.get("positive", sentiment.get("label_2", 0.0))
        surprise = emotions.get("surprise", 0.0)
        return min(1.0, (joy * 0.5 + positive * 0.3 + surprise * 0.2))

    def _calculate_uncertainty(
        self, headline: str, emotions: Dict[str, float]
    ) -> float:
        uncertainty_words = [
            "maybe",
            "might",
            "could",
            "possibly",
            "uncertain",
            "unclear",
            "unknown",
        ]
        word_score = (
            sum(1 for word in uncertainty_words if word in headline.lower()) * 0.2
        )

        fear = emotions.get("fear", 0.0)
        surprise = emotions.get("surprise", 0.0)

        return min(1.0, word_score + fear * 0.3 + surprise * 0.4)

    def _calculate_anxiety(self, emotions: Dict[str, float]) -> float:
        fear = emotions.get("fear", 0.0)
        anger = emotions.get("anger", 0.0)
        sadness = emotions.get("sadness", 0.0)

        return min(1.0, fear * 0.5 + anger * 0.3 + sadness * 0.2)

    def _calculate_confidence(
        self, emotions: Dict[str, float], sentiment: Dict[str, float]
    ) -> float:
        joy = emotions.get("joy", 0.0)
        positive = sentiment.get("positive", sentiment.get("label_2", 0.0))
        fear = emotions.get("fear", 0.0)

        confidence = (joy * 0.4 + positive * 0.4) - (fear * 0.2)
        return max(0.0, min(1.0, confidence))

    def _calculate_urgency(self, headline: str, emotions: Dict[str, float]) -> float:
        urgent_words = [
            "urgent",
            "breaking",
            "emergency",
            "crisis",
            "immediate",
            "now",
            "alert",
        ]
        word_score = sum(1 for word in urgent_words if word in headline.lower()) * 0.3

        fear = emotions.get("fear", 0.0)
        anger = emotions.get("anger", 0.0)

        return min(1.0, word_score + fear * 0.3 + anger * 0.2)


def analyze_sentiment(headline: str) -> str:
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_headline(headline)
    return json.dumps(results, indent=2)


def main():
    example_headline = "Stock market soars to record highs as investors show confidence"
    result = analyze_sentiment(example_headline)
    rprint(f"[bold green]Analyzing: [yellow]{example_headline}")
    rprint(f"[bold white]{result}")
    print("")

    with open("headlines.txt", "r") as f:
        headlines = f.readlines()
    for headline in headlines:
        headline = headline.strip()
        if headline:
            result = analyze_sentiment(headline)
            rprint(f"[bold green]Analyzing: [yellow]{headline}")
            rprint(f"[bold white]{result}")
            print("")


if __name__ == "__main__":
    main()
