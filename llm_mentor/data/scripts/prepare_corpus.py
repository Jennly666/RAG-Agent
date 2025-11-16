import json
from pathlib import Path
from typing import List

import pandas as pd
import tiktoken

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_CSV = ROOT_DIR / "data" / "interim" / "okx_trading_guide_raw.csv"
OUTPUT_CSV = ROOT_DIR / "data" / "processed" / "okx_trading_guide_chunks.csv"
OUTPUT_JSON = ROOT_DIR / "data" / "processed" / "okx_trading_guide_chunks.json"

MAX_TOKENS_PER_CHUNK = 500
MODEL_FOR_ENCODING = "gpt-3.5-turbo"


def count_tokens(text: str, model: str = MODEL_FOR_ENCODING) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    tokens_used = 0

    for p in paragraphs:
        t = count_tokens(p)
        if tokens_used + t > max_tokens and current:
            chunks.append("\n".join(current))
            current = [p]
            tokens_used = t
        else:
            current.append(p)
            tokens_used += t

    if current:
        chunks.append("\n".join(current))

    return chunks


def main():
    df_raw = pd.read_csv(RAW_CSV)

    df_raw = df_raw.head(5).copy()

    descriptions = [
        "Распространённые ошибки начинающих трейдеров и стратегии управления рисками",
        "Обзор торговых ботов OKX: виды, настройка и использование",
        "Способы обмена USDT без комиссии на платформе OKX",
        "Инструкция по покупке криптовалюты с банковской карты через P2P",
        "Интеграция TradingView с OKX: руководство по торговле и настройке",
    ]

    primary_tags = [
        "risk",
        "bots",
        "fees",
        "p2p",
        "tradingview",
    ]

    rows = []

    for _, row in df_raw.iterrows():
        article_id = row["id"]
        title = row["title"]
        url = row["url"]
        content = str(row["content"])

        idx = int(article_id)  # 0..4
        description = descriptions[idx]
        tag_primary = primary_tags[idx]

        chunks = chunk_text(content)

        for chunk_order, chunk in enumerate(chunks):
            rows.append(
                {
                    "article_id": article_id,
                    "chunk_id": f"{article_id}_{chunk_order}",
                    "title": title,
                    "url": url,
                    "description": description,
                    "tag_primary": tag_primary,
                    "lang": "ru",
                    "chunk_order": chunk_order,
                    "content": chunk,
                }
            )

    df_chunks = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_chunks.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Корпус подготовлен: {OUTPUT_CSV}")
    print(f"JSON сохранён: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
