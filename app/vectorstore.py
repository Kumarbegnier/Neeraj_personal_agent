import os
import logging
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


logger = logging.getLogger(__name__)

# Load env regardless of where uvicorn is started from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))




@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return embeddings implementation.

    Uses OpenAI embeddings with OPENAI_API_KEY from `.env`.
    """

    try:

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        dimensions: Optional[int] = None
        raw_dimensions = (os.getenv("OPENAI_EMBEDDING_DIMENSIONS") or "").strip()
        if raw_dimensions:
            try:
                dimensions = int(raw_dimensions)
            except ValueError as exc:
                raise RuntimeError("OPENAI_EMBEDDING_DIMENSIONS must be an integer") from exc


        proxy_url = (os.getenv("PROXY_URL") or "").strip()

        if proxy_url:
            # Use Proxy (recommended for team usage)
            embeddings: Embeddings = OpenAIEmbeddings(
                model=model,
                api_key="proxy-key",  # dummy key (proxy ignores)
                base_url=f"{proxy_url}/v1",
                dimensions=dimensions,
            )
        else:
            raise RuntimeError(
                "No embedding config found. Set PROXY_URL or OPENAI_API_KEY"
            )

        return embeddings
    
    except Exception:
        logger.exception("Embedding provider init failed")

        raise RuntimeError("Embedding provider unavailable")

def _env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value else None


def _require_env(name: str) -> str:
    value = _env(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def get_pinecone_vectorstore(namespace: Optional[str] = None):
    """Create a Pinecone-backed vector store.

    Namespace is intentionally a runtime argument so you can isolate embeddings per role
    (e.g. 'user', 'admin', 'partner', 'affiliate', 'sales').
    """

    try:
        from pinecone import Pinecone  # type: ignore
        from langchain_pinecone import PineconeVectorStore  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Pinecone dependencies not installed. Install 'pinecone' and 'langchain-pinecone'."
        ) from exc

    api_key = _require_env("PINECONE_API_KEY")
    index_name = _require_env("PINECONE_INDEX_NAME")
    text_key = os.getenv("PINECONE_TEXT_KEY", "text")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        text_key=text_key,
        namespace=namespace,
    )


def get_retriever(*, namespace: Optional[str], k: int = 4):
    vectorstore = get_pinecone_vectorstore(namespace=namespace)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_with_scores(*, namespace: Optional[str], query: str, k: int = 4):
    """Retrieve documents with similarity scores (best-effort).

    Returns: (docs, scores)
    - docs: list[Document]
    - scores: list[float] aligned with docs when available
    """

    vectorstore = get_pinecone_vectorstore(namespace=namespace)

    # PineconeVectorStore supports similarity_search_with_score.
    results = vectorstore.similarity_search_with_score(query, k=k)
    docs = [doc for (doc, _score) in results]
    scores = [float(_score) for (_doc, _score) in results]
    return docs, scores


def format_docs(docs) -> str:
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in (docs or []))


def upsert_texts(
    *,
    namespace: Optional[str],
    texts: list[str],
    metadatas: Optional[list[dict[str, Any]]] = None,
    ids: Optional[list[str]] = None,
):
    try :
        vectorstore = get_pinecone_vectorstore(namespace=namespace)

        if not vectorstore:
            logger.warning("Upsert skipped (no vectorstore)")
            return []


        return vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    except Exception:
        logger.exception("Upsert failed")
        return []