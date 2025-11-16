import sys
from pathlib import Path

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CONFIG  # noqa: E402

load_dotenv(ROOT_DIR / ".env")


def build_index():
    df = pd.read_csv(ROOT_DIR / CONFIG.processed_corpus_path)

    texts = df["content"].astype(str).tolist()
    metadatas = df[
        ["article_id", "chunk_id", "title", "description", "url", "tag_primary", "chunk_order"]
    ].to_dict(orient="records")

    embeddings = OpenAIEmbeddings(model=CONFIG.embedding_model)

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(ROOT_DIR / CONFIG.persist_directory),
    )

    vectordb.persist()
    print(f"Индекс построен и сохранён в {ROOT_DIR / CONFIG.persist_directory}")


if __name__ == "__main__":
    Path(ROOT_DIR / CONFIG.persist_directory).mkdir(parents=True, exist_ok=True)
    build_index()
