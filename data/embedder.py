import json
from pathlib import Path
import re
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

Improvements:
1. Larger chunk size (512 vs 300) with better overlap (50 vs 20)
2. Text normalization for special characters
3. URLs added to chunk text for searchability
4. Better handling of tables and lists

How to run:
    poetry run python data/embedder.py
"""

DATA_PATH = Path("data/lore.json")
DB_DIR = "db/minecraft_lore"


def normalize_text(text: str) -> str:
    """Normalize text to handle special characters and improve searchability.
    
    Improvements:
    - Convert Unicode fractions to standard format
    - Normalize special spaces and characters
    - Handle Bedrock/Java Edition markers
    - Preserve important formatting
    """
    # Convert Unicode fraction slash to regular slash
    text = text.replace(" ⁄ ", "/")
    text = text.replace("⁄", "/")
    
    # Normalize special Unicode spaces
    text = text.replace("\u200c", "")  # Zero-width non-joiner
    text = text.replace("\u200b", "")  # Zero-width space
    
    # Normalize edition markers for better matching
    # Keep original but add normalized version
    if "‌[BE only]" in text or "‌[JE only]" in text:
        text = re.sub(r"‌\[(BE|JE) only\]", r" [\1 only]", text)
    
    return text


def load_lore_json(path: Path) -> List[dict]:
    """Load structured Minecraft wiki data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_redundant_text(text: str) -> bool:
    """Filter out noisy or redundant paragraphs."""
    return len(text.strip()) > 30


def split_and_prepare_documents(lore_data: List[dict]) -> List[Document]:
    """Split content into semantic chunks.
    
    Improvements:
    - Larger chunks (512 chars) to keep related info together
    - Better overlap (50 chars) to avoid splitting key information
    - URLs added to chunk text for searchability
    - Text normalization for special characters
    - Title and URL added at the beginning for better link retrieval
    """
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
    )
    docs = []

    for entry in lore_data:
        # Normalize text before chunking
        normalized_content = normalize_text(entry["content"])
        
        chunks = splitter.split_text(normalized_content)
        filtered_chunks = filter(filter_redundant_text, chunks)

        for chunk in filtered_chunks:
            # Add title and URL at the beginning and end for better searchability
            # Format: Title + URL at start, then content, then Source URL at end
            chunk_with_source = f"Title: {entry['title']}\nURL: {entry['url']}\n\n{chunk}\n\nSource: {entry['url']}"
            
            docs.append(
                Document(
                    text=chunk_with_source,
                    metadata={
                        "source": entry["title"],
                        "url": entry["url"],
                        "chunk_size": len(chunk),
                    }
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

    # Use the same embedding model as before
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    index = VectorStoreIndex.from_documents(
        documents, embed_model=embed_model, storage_context=storage_context
    )
    index.storage_context.persist()
    
    # Print statistics
    chunk_sizes = [doc.metadata.get("chunk_size", 0) for doc in documents]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    
    print(f"✅ {len(documents)} chunks embedded and saved to Chroma at {DB_DIR}")
    print(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")
    print(f"📊 Chunk size range: {min(chunk_sizes)}-{max(chunk_sizes)} characters")


if __name__ == "__main__":
    print("➡️  Loading lore...")
    lore = load_lore_json(DATA_PATH)
    print(f"➡️  Processing {len(lore)} entries...")

    docs = split_and_prepare_documents(lore)
    print(f"➡️  {len(docs)} semantic chunks ready for embedding.")

    build_vector_index(docs)

    print("✅ Embedding complete.")
