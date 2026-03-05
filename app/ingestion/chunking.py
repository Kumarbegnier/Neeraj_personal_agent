from __future__ import annotations

from typing import Iterable


def chunk_texts(
    texts: Iterable[str],
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[str]:
    """Chunk long texts for embeddings.

    Notes:
    - Uses LangChain's RecursiveCharacterTextSplitter if available.
    - Falls back to a simple splitter if LangChain text splitters aren't available.
    """

    texts_list = [t for t in texts if t and t.strip()]
    if not texts_list:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text("\n\n".join(texts_list))
    except Exception:
        # Simple fallback: fixed windows.
        joined = "\n\n".join(texts_list)
        step = max(1, chunk_size - chunk_overlap)
        chunks: list[str] = []
        for start in range(0, len(joined), step):
            chunk = joined[start : start + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
