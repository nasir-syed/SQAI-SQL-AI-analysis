from __future__ import annotations

from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer
import chromadb

class VectorStore:
    
    # Manages vector embedding and storage using Chroma DB.
    
    # Using HuggingFace sentence-transformers for semantic similarity
    # Stores documents locally in Chroma's persistent database.

    def __init__(
        self,
        collection_name: str = "schema_tables",
        persist_dir: str = "./schema_embeddings",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:

        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        
        # Initialize sentence-transformers model for embeddings
        self.embeddings_model = SentenceTransformer(embedding_model)
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )

    def add_documents(self, documents: Dict[str, Dict[str, Any]]) -> None:

        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for table_name, doc in documents.items():
            # Generate embedding
            embedding = self._embed_text(doc["content"])
            
            ids.append(table_name)
            embeddings.append(embedding)
            documents_text.append(doc["content"])
            metadatas.append(doc["metadata"])
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents_text,
            metadatas=metadatas,
        )

    def retrieve_relevant_documents(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
 
        # Embed the query
        query_embedding = self._embed_text(query)
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        # Format results
        retrieved_docs = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = {
                    "table_name": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                }
                retrieved_docs.append(doc)
        
        return retrieved_docs

    def delete_documents(self, table_names: List[str]) -> None:
        if table_names:
            self.collection.delete(ids=table_names)

    def delete_all_documents(self) -> None:
        # For full reprocessing
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass  
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_document_count(self) -> int:
        return self.collection.count()

    def list_all_tables(self) -> List[str]:
        try:
            results = self.collection.get(include=[])
            return results.get("ids", []) if results else []
        except Exception:
            return []

    def _embed_text(self, text: str) -> List[float]:
        embedding = self.embeddings_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def persist(self) -> None:
        try:
            self.chroma_client.persist()
        except Exception:
            pass


class VectorStoreManager:
    
    # High-level manager for vector store operations.
    
    # Wraps VectorStore and provides convenience methods for:
    # -> Indexing schema documents
    # -> Retrieving relevant tables
    # -> Managing lifecycle (sync, persist, etc.)
    

    def __init__(
        self,
        persist_dir: str = "./schema_embeddings",
    ) -> None:
        self.vector_store = VectorStore(
            persist_dir=persist_dir,
        )

    def index_schema_documents(
        self,
        documents: Dict[str, Dict[str, Any]],
    ) -> None:
        self.vector_store.add_documents(documents)
        self.vector_store.persist()

    def get_relevant_schemas(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:

        return self.vector_store.retrieve_relevant_documents(query, top_k=top_k)

    def format_retrieved_for_llm(
        self,
        retrieved_docs: List[Dict[str, Any]],
    ) -> str:

        if not retrieved_docs:
            return "No relevant tables found."
        
        lines = ["RETRIEVED TABLE SCHEMAS:"]
        lines.append("=" * 50)
        
        for i, doc in enumerate(retrieved_docs, 1):
            lines.append(f"\n[Table {i}: {doc['metadata'].get('full_table_name', 'Unknown')}]")
            lines.append(f"Rows: {doc['metadata'].get('row_count', 'Unknown'):,}")
            lines.append("-" * 50)
            lines.append(doc["content"])
            lines.append("")
        
        return "\n".join(lines)

    def sync_with_schema(
        self,
        documents: Dict[str, Dict[str, Any]],
        existing_tables: List[str],
    ) -> Dict[str, Any]:
        current_tables = set(documents.keys())
        indexed_tables = set(existing_tables)
        
        # Find changes
        new_tables = current_tables - indexed_tables
        removed_tables = indexed_tables - current_tables
        updated_tables = current_tables & indexed_tables  # All existing get re-indexed
        
        # Apply changes
        if removed_tables:
            self.vector_store.delete_documents(list(removed_tables))
        
        # Re-index all documents (update + new)
        if documents:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
        
        return {
            "added": len(new_tables),
            "removed": len(removed_tables),
            "updated": len(updated_tables),
            "total": len(current_tables),
        }
