from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
