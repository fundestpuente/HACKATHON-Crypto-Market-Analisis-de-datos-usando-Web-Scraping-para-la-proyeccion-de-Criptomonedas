import pandas as pd

from app.sentiment.scraper import scrape_coindesk_headlines
from app.sentiment.analyzer import analyze_sentiment


def run_pipeline():
    news = scrape_coindesk_headlines(limit=30)

    enriched_news = []

    for item in news:
        sentiment = analyze_sentiment(item["title"])

        enriched_news.append({
            "date": item["date"],
            "source": item["source"],
            "title": item["title"],
            "sentiment_score": sentiment["polarity"],
            "sentiment_label": sentiment["label"]
        })

    df = pd.DataFrame(enriched_news)

    df.to_csv("data/sentiment_coindesk.csv", index=False)

    return df


if __name__ == "__main__":
    df_sentiment = run_pipeline()
    print(df_sentiment.head())
