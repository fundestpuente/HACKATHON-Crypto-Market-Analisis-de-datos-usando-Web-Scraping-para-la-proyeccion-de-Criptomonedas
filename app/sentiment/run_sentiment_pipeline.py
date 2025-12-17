from pathlib import Path

import pandas as pd

from app.config import settings
from app.sentiment.analyzer import analyze_sentiment
from app.sentiment.scraper import scrape_coindesk_headlines


def run_pipeline(limit: int = 30, output_path: Path | None = None) -> pd.DataFrame:
    news = scrape_coindesk_headlines(limit=limit)

    enriched_news = []

    for item in news:
        sentiment = analyze_sentiment(item["title"])

        enriched_news.append(
            {
                "date": item["date"],
                "source": item["source"],
                "title": item["title"],
                "sentiment_score": sentiment["polarity"],
                "sentiment_label": sentiment["label"],
            }
        )

    df = pd.DataFrame(enriched_news)

    output_path = output_path or (settings.data_dir / "sentiment_coindesk.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    df_sentiment = run_pipeline()
    print(df_sentiment.head())
