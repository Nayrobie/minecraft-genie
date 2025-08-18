import json
from pathlib import Path
from typing import List
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import chromadb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

"""
Transforms the website content from lore.json into chunked embeddings and stores them in Chroma.

What it does:
- Loads JSON with 'title', 'url', 'content' fields
- Cleans and filters the content
- Splits into semantic chunks
- Embeds using OpenAI
- Stores the embeddings into Chroma DB

How to run:
    poetry run python data/embedder.py
"""

DATA_PATH = Path("data/lore.json")
DB_DIR = "db/minecraft_lore"


def load_lore_json(path: Path) -> List[dict]:
    """Load structured Minecraft wiki data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_redundant_text(text: str) -> bool:
    """Filter out noisy or redundant paragraphs."""
    return len(text.strip()) > 20


def split_and_prepare_documents(lore_data: List[dict]) -> List[Document]:
    """Split content into semantic chunks, associate metadata.

    Optionally, table rows (like bullet points or lists) can be treated as separate chunks
    for improved search granularity.
    """
    splitter = SentenceSplitter(chunk_size=300, chunk_overlap=20)
    docs = []

    for entry in lore_data:
        chunks = splitter.split_text(entry["content"])
        filtered_chunks = filter(filter_redundant_text, chunks)

        for chunk in filtered_chunks:
            docs.append(
                Document(
                    text=chunk, metadata={"source": entry["title"], "url": entry["url"]}
                )
            )
    return docs


def build_vector_index(documents: List[Document]) -> VectorStoreIndex:
    """Build and persist Chroma vector index."""
    # Create the database directory if it doesn't exist
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_DIR)

    # Get or create collection
    collection = client.get_or_create_collection("minecraft_lore")

    # Create vector store with explicit collection
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, embed_model=OpenAIEmbedding(), storage_context=storage_context
    )
    index.storage_context.persist()
    print(f"✅ {len(documents)} chunks embedded and saved to Chroma at {DB_DIR}")


if __name__ == "__main__":
    print("➡️ Loading lore...")
    lore = load_lore_json(DATA_PATH)
    print(f"➡️  Processing {len(lore)} entries...")

    docs = split_and_prepare_documents(lore)
    print(f"➡️ {len(docs)} semantic chunks ready for embedding.")

    build_vector_index(docs)
    
    print("✅ Embedding complete.")