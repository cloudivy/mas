import numpy as np
import openai
import re


def get_embedding(api_key: str, text: str) -> list:
    """Return embedding vector for a text string."""
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:1800],
    )
    return response.data[0].embedding


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two embedding vectors."""
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 1.0
    return float(np.dot(va, vb) / denom)


def jaccard_terms(a: str, b: str) -> float:
    """Jaccard similarity on 4+ character word tokens."""
    tok = lambda s: set(re.findall(r'\b\w{4,}\b', s.lower()))
    sa, sb = tok(a), tok(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
