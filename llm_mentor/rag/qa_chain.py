import json
from typing import List

from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

from config import CONFIG
from rag.prompts import qa_prompt, self_check_prompt


def build_qa_chain():
    llm = ChatOpenAI(model_name=CONFIG.chat_model, temperature=0)
    return LLMChain(llm=llm, prompt=qa_prompt)


def build_self_check_chain():
    llm = ChatOpenAI(model_name=CONFIG.chat_model, temperature=0)
    return LLMChain(llm=llm, prompt=self_check_prompt)


def extract_statements(answer: str) -> List[str]:
    lines = []
    capture = False
    for line in answer.splitlines():
        low = line.strip().lower()
        if low.startswith("подробно"):
            capture = True
            continue
        if capture:
            if low.startswith("источники"):
                break
            if line.strip():
                lines.append(line.strip())
    return lines


def filter_answer_with_self_check(answer: str, context: str, self_check_chain: LLMChain) -> str:
    statements = extract_statements(answer)
    if not statements:
        return answer

    statements_text = "\n".join(f"- {s}" for s in statements)
    result = self_check_chain.run({"statements": statements_text, "context": context})

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return answer

    allowed_statements = [item["statement"] for item in parsed if item.get("label") == "TRUE"]

    if not allowed_statements:
        return (
            "Кратко:\n"
            "- Контекст не подтвердил детали ответа.\n\n"
            "Подробно:\n"
            "Модель не нашла в материалах OKX явного подтверждения деталей вопроса. "
            "Лучше обратиться к статьям напрямую.\n\n"
            "Источники:\n"
            "- см. контекст запроса."
        )

    # вставляем только TRUE-утверждения в блок "Подробно"
    rebuilt = []
    in_detailed = False
    for line in answer.splitlines():
        low = line.strip().lower()
        if low.startswith("подробно"):
            in_detailed = True
            rebuilt.append(line)
            for s in allowed_statements:
                rebuilt.append(f"- {s}")
            continue
        if in_detailed:
            if low.startswith("источники"):
                in_detailed = False
                rebuilt.append(line)
            # пропускаем старые подробности
            continue
        rebuilt.append(line)

    return "\n".join(rebuilt)
