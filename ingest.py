from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

load_dotenv()

SUPPORTED_LOADERS = {
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
}


def load_documents(directory: Path) -> list:
    documents = []
    for path in sorted(directory.iterdir()):
        loader_factory = SUPPORTED_LOADERS.get(path.suffix.lower())
        if loader_factory is None or not path.is_file():
            continue
        print(f"Loading {path.name}...")
        documents.extend(loader_factory(str(path)).load())
    return documents


def sanitize_documents(documents: list) -> list:
    for document in documents:
        document.page_content = document.page_content.strip()
    return documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def ingest_data(directory_path: str = "./data") -> None:
    directory = Path(directory_path)
    print(f"--- Step 1: Identify data sources in {directory} ---")

    documents = load_documents(directory)
    if not documents:
        print("No supported files found in the data directory.")
        return

    print(f"\n--- Step 2: Prepare and Sanitize data ({len(documents)} documents loaded) ---")
    sanitize_documents(documents)

    print("\n--- Step 3: Chunk the data ---")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("\n--- Step 4: Create embeddings and Store the data ---")
    mongo_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "ai_agent_db")
    collection_name = os.getenv("MONGODB_COLLECTION_NAME", "vector_embeddings")
    index_name = os.getenv("MONGODB_VECTOR_INDEX_NAME", "vector_index")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if not mongo_uri or not os.getenv("OPENAI_API_KEY"):
        print("Error: MONGODB_URI or OPENAI_API_KEY not found in .env file.")
        return

    client = MongoClient(mongo_uri)
    try:
        collection = client[db_name][collection_name]
        embeddings = OpenAIEmbeddings(model=embedding_model)

        MongoDBAtlasVectorSearch.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection=collection,
            index_name=index_name,
        )
    finally:
        client.close()

    print(f"Successfully stored {len(chunks)} chunks in MongoDB collection '{collection_name}'.")


def main() -> None:
    data_directory = Path("./data")
    if not data_directory.exists():
        data_directory.mkdir(parents=True, exist_ok=True)
        print("Created './data' directory. Please place your files there and run again.")
        return

    ingest_data(str(data_directory))


if __name__ == "__main__":
    main()
