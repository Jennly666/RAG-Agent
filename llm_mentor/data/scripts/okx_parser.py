import csv
import time
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parents[2]

BASE_URL = "https://www.okx.com"
INDEX_HTML_PATH = ROOT_DIR / "data" / "raw" / "okx_index.html"
OUTPUT_CSV_PATH = ROOT_DIR / "data" / "interim" / "okx_trading_guide_raw.csv"

HEADERS = {
    "Referer": "https://www.okx.com/ru/learn/tag/beginner-trading-guide",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0.0.0 Safari/537.36"
    ),
}


def parse_index_page():
    with INDEX_HTML_PATH.open("r", encoding="utf-8") as f:
        src = f.read()
    soup = BeautifulSoup(src, "lxml")
    return soup.find_all("div", class_="index_postItem__DW1Rb")


def fetch_article_content(relative_url: str) -> str:
    url = BASE_URL + relative_url
    html = requests.get(url=url, headers=HEADERS).text
    soup = BeautifulSoup(html, "lxml")
    pattern = re.compile(r"index_articleMainContent__\w+")
    div = soup.find("div", class_=pattern)
    return div.get_text(separator="\n", strip=True)


def main():
    groups = parse_index_page()
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "url", "content"])

        for idx, group in enumerate(groups):
            title = group.find("h3", class_="index_title__0XdIR").text.strip()
            link = group.find("a")["href"]
            content = fetch_article_content(link)

            writer.writerow([idx, title, BASE_URL + link, content])
            print(f"Parsed article {idx}: {title}")
            time.sleep(3)

    print(f"\nГотово! Данные сохранены в {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
