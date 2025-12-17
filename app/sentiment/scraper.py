import requests
from bs4 import BeautifulSoup
from datetime import datetime


COINDESK_URL = "https://www.coindesk.com/"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def scrape_coindesk_headlines(limit=20):
    response = requests.get(COINDESK_URL, timeout=10, headers=DEFAULT_HEADERS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    articles = soup.find_all("a", href=True)

    for article in articles:
        title = article.get_text(strip=True)

        if title and len(title.split()) > 4:
            headlines.append({
                "title": title,
                "source": "CoinDesk",
                "date": datetime.utcnow().date().isoformat()
            })

        if len(headlines) >= limit:
            break

    return headlines


if __name__ == "__main__":
    news = scrape_coindesk_headlines()
    for n in news:
        print(n)
