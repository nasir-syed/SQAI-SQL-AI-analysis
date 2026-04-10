from __future__ import annotations

from typing import Any, Dict, List, Optional

from vector_store import VectorStoreManager

# High-level interface for retrieving relevant table schemas using vector similarity
class SchemaRetriever:

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        default_top_k: int = 3,
    ) -> None:

        self.vector_mgr = vector_store_manager
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        if top_k is None:
            top_k = self.default_top_k
        
        retrieved_docs = self.vector_mgr.get_relevant_schemas(query, top_k=top_k)
        
        return retrieved_docs

    def retrieve_with_fallback(
        self,
        query: str,
        top_k: Optional[int] = None,
        fallback_k: int = 5,
    ) -> Dict[str, Any]:
        try:
            docs = self.retrieve(query, top_k=top_k)
            
            if docs:
                return {
                    "documents": docs,
                    "source": "vector",
                    "count": len(docs),
                }
            else:
                # Vector store might be empty
                return {
                    "documents": [],
                    "source": "empty",
                    "count": 0,
                    "warning": "Vector store is empty. Run schema sync first.",
                }
        except Exception as e:
            return {
                "documents": [],
                "source": "error",
                "count": 0,
                "error": str(e),
            }

    def format_for_llm(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> str:
        if not retrieved_docs:
            return "No relevant table schemas could be retrieved for the query."
        
        lines = []
        lines.append("=" * 70)
        lines.append("RETRIEVED DATABASE SCHEMA")
        lines.append("=" * 70)
        lines.append(f"\nQuery: {query}\n")
        lines.append("The following tables are relevant to your query:\n")
        
        for idx, doc in enumerate(retrieved_docs, 1):
            table_name = doc["metadata"].get("full_table_name", "Unknown")
            row_count = doc["metadata"].get("row_count", "Unknown")
            cols = doc["metadata"].get("column_count", "?")
            
            lines.append(f"\n───────────────────────────────────────────────────────────────────────")
            lines.append(f"TABLE {idx}: {table_name}")
            lines.append(f"Columns: {cols} | Approximate Rows: {row_count:,}")
            lines.append(f"───────────────────────────────────────────────────────────────────────\n")
            lines.append(doc["content"])
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)

    def format_for_json_schema(
        self,
        retrieved_docs: List[Dict[str, Any]],
    ) -> str:
        # Format retrieved documents as JSON schema for stricter LLM parsing.
        import json
        
        if not retrieved_docs:
            return json.dumps({"error": "No tables found"})
        
        schemas = []
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            
            schema_obj = {
                "table_name": metadata.get("full_table_name", ""),
                "row_count": metadata.get("row_count", 0),
                "schema": metadata.get("schema_name", ""),
                "columns": len(metadata.get("column_count", 0)),
                "summary": doc["content"].split("\n")[0:5],  
            }
            schemas.append(schema_obj)
        
        return json.dumps(
            {"tables": schemas},
            indent=2,
            default=str,
        )

    def debug_info(self) -> Dict[str, Any]:
        try:
            count = self.vector_mgr.vector_store.get_document_count()
            tables = self.vector_mgr.vector_store.list_all_tables()
            return {
                "vector_store_ready": True,
                "document_count": count,
                "tables": tables,
            }
        except Exception as e:
            return {
                "vector_store_ready": False,
                "error": str(e),
            }
