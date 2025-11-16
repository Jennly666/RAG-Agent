import re

FORBIDDEN_PATTERNS = [
    r"обойти\s+kyc",
    r"обойти\s+верификац",
    r"обойти\s+aml",
    r"инсайд",
    r"внутренн(ая|ую)\s+информац",
    r"гарантирова[нт]\s+прибыл",
    r"обмануть\s+(бирж|систем|пользоват)",
]


def is_blocked_query(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in FORBIDDEN_PATTERNS)
