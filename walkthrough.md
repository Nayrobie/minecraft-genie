cmd shift v

# Minecraft Genie

## Web scrapping and embedding
![alt text](image.png)
### Document loading & Preprocessing (LlamaIndex Python SDK)
- Gather Minecraft documentation from [https://minecraft.wiki/](https://minecraft.wiki/) into `lore_docs.txt`
- One brief topic per chunk

### Index creation (LlamaIndex & Chroma backend)
- Embed and save into vector database with the `embedder.py` script using Chroma  
  - **Chunking**: Splitting large texts into smaller parts (e.g. 100–300 tokens)
  - **Embeddings**: Transform text into numerical vectors for similarity search  
  - **Vector DB**: Stores those vectors and lets you search by similarity  

- Load the vectors using `retriever.py` to perform the semantic search

### LangGraph workflow
- Set up LangGraph in `graph.py` to orchestrate the flow  
  - **Node 1**: Retriever node (calls LlamaIndex retriever)  
  - **Node 2**: LLM Answer node (sends prompt + context to OpenAI)