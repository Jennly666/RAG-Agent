from dataclasses import dataclass


@dataclass
class AppConfig:
    persist_directory: str = "chroma_db"
    processed_corpus_path: str = "data/processed/okx_trading_guide_chunks.csv"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-3.5-turbo"
    max_context_tokens: int = 3000
    score_threshold: float = 0.3  # порог релевантности


CONFIG = AppConfig()