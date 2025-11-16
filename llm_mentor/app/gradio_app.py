import sys
from pathlib import Path
import asyncio

import gradio as gr
from dotenv import load_dotenv

# app -> root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from config import CONFIG
from rag.retriever import build_compression_retriever, async_retrieve_docs
from rag.qa_chain import (
    build_qa_chain,
    build_self_check_chain,
    filter_answer_with_self_check,
)
from app.safety import is_blocked_query


retriever = build_compression_retriever()
qa_chain = build_qa_chain()
self_check_chain = build_self_check_chain()


async def answer_question_async(query: str) -> str:
    docs = await async_retrieve_docs(query, retriever, max_tokens=CONFIG.max_context_tokens)

    if not docs:
        return (
            f"**Вопрос:** {query}\n\n"
            "**Ответ:**\n"
            "Я не нашёл релевантной информации в материалах OKX, которые у меня есть. "
            "Не могу ответить без домыслов."
        )

    context = "\n\n".join([d.page_content for d in docs])
    raw_answer = qa_chain.run({"context": context, "query": query})

    final_answer = filter_answer_with_self_check(raw_answer, context, self_check_chain)

    # собираем уникальные источники
    seen = set()
    source_lines = []
    for d in docs:
        title = d.metadata.get("title")
        desc = d.metadata.get("description")
        key = (title, desc)
        if key in seen:
            continue
        seen.add(key)
        source_lines.append(f"- {title} — {desc}")

    sources = "\n".join(source_lines)

    return (
        f"**Вопрос:** {query}\n\n"
        f"{final_answer}\n\n"
        f"---\n"
        f"**Источники (по данным RAG):**\n{sources}"
    )


def sync_wrapper(query: str) -> str:
    if not query.strip():
        return "Введите, пожалуйста, вопрос."

    if is_blocked_query(query):
        return (
            "**Вопрос отклонён по соображениям безопасности.**\n\n"
            "Я не могу помогать с обходом KYC/AML, получать инсайдерскую информацию "
            "или давать гарантии прибыли. Попробуйте переформулировать вопрос."
        )

    return asyncio.run(answer_question_async(query))


def show_loading(_query: str):
    return gr.update(value="CryptoMentor думает...", visible=True)


def answer_and_hide_status(query: str):
    answer = sync_wrapper(query)
    return (
        gr.update(value=answer),
        gr.update(visible=False),
    )


def clear_all():
    """Сброс полей."""
    return "", "", gr.update(value="", visible=False)


def main():
    theme = gr.themes.Soft()

    css = """
    #main-container {
        max-width: 900px;
        margin: 0 auto;
        padding-top: 2rem;
    }
    """

    with gr.Blocks(
        title="CryptoMentor — OKX Helper",
        theme=theme,
        css=css,
    ) as demo:
        with gr.Column(elem_id="main-container"):
            gr.Markdown(
                """
# CryptoMentor — эксперт по OKX

Задайте вопрос про трейдинг и сервисы OKX, и нейро-сотрудник ответит, опираясь на статьи OKX Academy.
"""
            )

            query_input = gr.Textbox(
                label="Введите вопрос",
                placeholder="Например: Какие ошибки совершают новички в криптотрейдинге?",
                lines=2,
            )

            with gr.Row():
                submit_btn = gr.Button("Спросить CryptoMentor", variant="primary")
                clear_btn = gr.Button("Очистить")

            gr.Examples(
                examples=[
                    ["Какие ошибки совершают новички в криптотрейдинге?"],
                    ["Как настроить торгового бота на OKX?"],
                    ["Как купить USDT без комиссии на OKX?"],
                    ["Как работает P2P-покупка с банковской карты на OKX?"],
                ],
                inputs=query_input,
            )

            status_md = gr.Markdown("", visible=False)
            output_box = gr.Markdown(label="Ответ")

        # индикатор загрузки
        submit_btn.click(
            fn=show_loading,
            inputs=query_input,
            outputs=status_md,
            queue=False,
        )

        submit_btn.click(
            fn=answer_and_hide_status,
            inputs=query_input,
            outputs=[output_box, status_md],
            queue=True,
        )

        clear_btn.click(
            fn=clear_all,
            inputs=None,
            outputs=[query_input, output_box, status_md],
            queue=False,
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
