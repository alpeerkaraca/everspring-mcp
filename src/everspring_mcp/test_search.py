import asyncio
from everspring_mcp.vector.config import VectorConfig
from everspring_mcp.vector.chroma_client import ChromaClient
from everspring_mcp.vector.embeddings import Embedder

async def search(query: str, n_results: int = 5):
     config = VectorConfig.from_env()
     chroma = ChromaClient(config)
     embedder = Embedder(model_name=config.embedding_model)

     vectors = await embedder.embed_texts([query])
     results = chroma.query(query_embeddings=vectors, n_results=n_results)

     for i, (doc, meta, dist) in enumerate(zip(
         results['documents'][0],
         results['metadatas'][0],
         results['distances'][0]
     )):
         print(f"[{i+1}] {meta.get('title')} (dist: {dist:.3f})")
         print()

while True:
     user_input = input("Enter your search query (or 'exit' to quit): ")
     if user_input.lower() == 'exit':
         break
     asyncio.run(search(user_input))
