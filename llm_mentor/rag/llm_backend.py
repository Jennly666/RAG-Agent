from typing import List, Dict

from langchain_openai import ChatOpenAI


class LLMBackend:
    """
    Обёртка над LLM-провайдерами.
    Сейчас реализован только OpenAI, но интерфейс позволяет добавить локальную LLM.
    """

    def __init__(self, provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        self.provider = provider
        self.model_name = model_name

        if provider == "openai":
            self.client = ChatOpenAI(model_name=model_name, temperature=0)
        elif provider == "local":
            # Здесь можно реализовать обращение к локальной LLM по HTTP
            raise NotImplementedError("LOCAL backend is not implemented yet.")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.provider == "openai":
            resp = self.client.invoke(messages)
            return resp.content
        else:
            raise NotImplementedError
