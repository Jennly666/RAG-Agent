from typing import List

import tiktoken
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from config import CONFIG
from rag.prompts import compression_prompt


def count_tokens(text: str, model: str = CONFIG.chat_model) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def truncate_docs(docs, query: str, max_tokens: int) -> List:
    allowed = []
    used = count_tokens(query)
    for d in docs:
        t = count_tokens(d.page_content)
        if used + t <= max_tokens:
            allowed.append(d)
            used += t
        else:
            break
    return allowed


def build_base_retriever():
    vectordb = Chroma(
        persist_directory=CONFIG.persist_directory,
        embedding_function=OpenAIEmbeddings(model=CONFIG.embedding_model),
    )

    base_retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    return base_retriever


def build_compression_retriever():
    base_retriever = build_base_retriever()
    compressor_llm = ChatOpenAI(model_name=CONFIG.chat_model, temperature=0)

    compressor = LLMChainExtractor.from_llm(
        llm=compressor_llm,
        prompt=compression_prompt,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )
    return compression_retriever


async def async_retrieve_docs(query: str, retriever, max_tokens: int = CONFIG.max_context_tokens):
    docs = await retriever.aget_relevant_documents(query)
    safe_docs = truncate_docs(docs, query, max_tokens=max_tokens)
    return safe_docs
