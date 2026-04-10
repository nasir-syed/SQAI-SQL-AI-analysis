from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schema_management.schema_inspector import SchemaInspector
from schema_management.schema_document_generator import SchemaDocumentGenerator
from vector_store import VectorStoreManager

# Manages schema change detection and incremental vector index updates.
class SchemaSynchronizer:

    # Default location for storing metadata snapshots
    DEFAULT_SNAPSHOT_FILE = "./schema_metadata_snapshot.json"

    def __init__(
        self,
        schema_inspector: SchemaInspector,
        doc_generator: SchemaDocumentGenerator,
        vector_store_manager: VectorStoreManager,
        snapshot_file: str = DEFAULT_SNAPSHOT_FILE,
    ) -> None:
        self.inspector = schema_inspector
        self.doc_gen = doc_generator
        self.vector_mgr = vector_store_manager
        self.snapshot_file = snapshot_file

    def sync(self, force_full_reindex: bool = False) -> Dict[str, Any]:

        # Synchronize the vector index with the current database schema.
            
        # Flow:
        #   1. Extract current database schema
        #   2. Build current metadata from schema structure (no document generation)
        #   3. Load previous metadata snapshot
        #   4. Detect changes based on:
        #      - New tables detected
        #      - Existing tables removed
        #      - Structural changes: columns added/removed/modified
        #   5. Generate documents ONLY for tables with structural changes
        #   6. Apply changes to vector store (add new, delete removed, update modified)
        #   7. Save new metadata snapshot

        try:
            # Get current database schema
            current_schema = self.inspector.extract()
            self.doc_gen.schema_data = current_schema
            
            # Build metadata from schema structure (lightweight, no document generation)
            current_metadata = self._build_metadata_from_schema(current_schema)
            
            # Load previous snapshot
            previous_metadata = self._load_snapshot()
            
            # Detect changes based on schema structure
            if force_full_reindex or not previous_metadata:
                # Full reprocessing
                changes = self._detect_full_reindex_needed(current_metadata)
            else:
                # Granular change detection comparing table structure
                changes = self._detect_changes(current_metadata, previous_metadata)
            
            # Generate documents ONLY for changed tables
            changed_documents = self._generate_only_changed_documents(
                current_schema, changes
            )
            
            # Apply changes to vector store
            if changes["tables_added"] > 0 or changes["tables_removed"] > 0 or changes["tables_modified"] > 0:
                self._apply_changes(changed_documents, changes)
            
            # Save new metadata snapshot
            self._save_snapshot(current_metadata)
            
            total_tables = len(current_metadata)
            unchanged = total_tables - changes["tables_added"] - changes["tables_removed"] - changes["tables_modified"]
            
            return {
                "success": True,
                "tables_added": changes["tables_added"],
                "tables_removed": changes["tables_removed"],
                "tables_modified": changes["tables_modified"],
                "tables_unchanged": max(0, unchanged),
                "total_tables": total_tables,
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def _build_metadata_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        # Extract metadata directly from raw schema without full document generation.

        metadata = {}
        target_schema = schema.get("target_database", "")
        
        tables = schema.get("tables", {}).get(target_schema, [])
        columns = schema.get("columns", {})
        primary_keys = schema.get("primary_keys", {})
        foreign_keys = schema.get("foreign_keys", [])
        
        for table in tables:
            table_name = table["name"]
            full_table_name = f"{target_schema}.{table_name}"
            
            # Get columns for this table
            table_columns = columns.get(full_table_name, [])
            
            # Get primary keys for this table
            table_pks = primary_keys.get(full_table_name, [])
            
            # Get foreign keys referencing this table
            table_fks = [
                fk for fk in foreign_keys 
                if fk.get("table") == table_name
            ]
            
            # Build minimal metadata for change detection
            # This preserves table structure without embedding content
            metadata[full_table_name] = {
                "name": table_name,
                "schema": target_schema,
                "columns": [
                    {
                        "name": col.get("name", ""),
                        "type": col.get("type", ""),
                        "nullable": col.get("nullable", True),
                        "key": col.get("key", ""),
                    }
                    for col in table_columns
                ],
                "primary_keys": table_pks,
                "foreign_keys": table_fks,
                "comment": table.get("comment", ""),
                "row_count": table.get("approx_rows", 0),
            }
        
        return metadata

    def _detect_full_reindex_needed(
        self,
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Detect when full reindexing is needed (no previous snapshot or corrupted index).

        return {
            "tables_added": len(current),
            "tables_removed": 0,
            "tables_modified": 0,
            "added_list": list(current.keys()),
            "removed_list": [],
            "modified_list": [],
        }

    def _detect_changes(
        self,
        current: Dict[str, Any],
        previous: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        # Detect schema changes between current and previous metadata.
        
        current_tables = set(current.keys())
        previous_tables = set(previous.keys())
        
        added = list(current_tables - previous_tables)
        removed = list(previous_tables - current_tables)
        
        # Check for modifications in existing tables
        modified = []
        for table_name in current_tables & previous_tables:
            if self._has_table_changed(
                current[table_name], 
                previous.get(table_name, {})
            ):
                modified.append(table_name)
        
        return {
            "tables_added": len(added),
            "tables_removed": len(removed),
            "tables_modified": len(modified),
            "added_list": added,
            "removed_list": removed,
            "modified_list": modified,
        }

    def _has_table_changed(
        self,
        current_meta: Dict[str, Any],
        previous_meta: Dict[str, Any],
    ) -> bool:
        
        # Determine if a table's schema has structurally changed.

        # Get column definitions
        current_cols = current_meta.get("columns", [])
        previous_cols = previous_meta.get("columns", [])
        
        # if different column count, then change
        if len(current_cols) != len(previous_cols):
            return True
        
        # Map columns by name for comparison
        current_col_defs = {col["name"]: col for col in current_cols}
        previous_col_defs = {col["name"]: col for col in previous_cols}
        
        # Check for added/removed columns
        current_col_names = set(current_col_defs.keys())
        previous_col_names = set(previous_col_defs.keys())
        
        if current_col_names != previous_col_names:
            # Column was added or removed
            return True
        
        # Check each column for type/nullable changes
        for col_name in current_col_names:
            cur_col = current_col_defs[col_name]
            prev_col = previous_col_defs[col_name]
            
            # Type change = structural change
            if cur_col.get("type") != prev_col.get("type"):
                return True
            
            # Nullability change = structural change
            if cur_col.get("nullable") != prev_col.get("nullable"):
                return True
            
            # Key status change (became PK, lost PK, etc)
            if cur_col.get("key") != prev_col.get("key"):
                return True
        
        # Check primary key constraint changes
        current_pks = current_meta.get("primary_keys", [])
        previous_pks = previous_meta.get("primary_keys", [])
        if current_pks != previous_pks:
            return True
        
        # Check foreign key relationship changes
        current_fks = current_meta.get("foreign_keys", [])
        previous_fks = previous_meta.get("foreign_keys", [])
        if current_fks != previous_fks:
            return True
        
        # No structural changes detected
        return False

    def _generate_only_changed_documents(
        self,
        schema: Dict[str, Any],
        changes: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        
        # Generate documents ONLY for tables that have changed.
        
        documents = {}
        target_schema = schema.get("target_database", "")
        
        # Only regenerate for newly added and modified tables
        # Unchanged tables skip document generation entirely

        tables_to_regenerate = changes.get("added_list", []) + changes.get("modified_list", [])
        
        if not tables_to_regenerate:
            # No changes detected, return empty dict (no documents to update)
            return documents
        
        tables = schema.get("tables", {}).get(target_schema, [])
        table_name_map = {table["name"]: table for table in tables}
        
        for full_table_name in tables_to_regenerate:
            short_table_name = full_table_name.split(".")[-1] if "." in full_table_name else full_table_name
            
            if short_table_name in table_name_map:
                try:
                    doc = self.doc_gen.generate_table_document(short_table_name, target_schema)
                    documents[full_table_name] = doc
                except Exception as e:
                    print(f"Warning: Could not generate document for {full_table_name}: {e}")
        
        return documents

    def _apply_changes(
        self,
        documents: Dict[str, Dict[str, Any]],
        changes: Dict[str, Any],
    ) -> None:
        # Apply detected changes to the vector store.
        
        # Delete removed tables from vector store
        if changes["removed_list"]:
            self.vector_mgr.vector_store.delete_documents(changes["removed_list"])
        
        # Re-index modified and newly added tables
        docs_to_index = {}
        for table_name in changes["added_list"] + changes["modified_list"]:
            if table_name in documents:
                docs_to_index[table_name] = documents[table_name]
        
        if docs_to_index:
            self.vector_mgr.vector_store.add_documents(docs_to_index)
            self.vector_mgr.vector_store.persist()

    def _load_snapshot(self) -> Dict[str, Any]:
        
        # Load the previous metadata snapshot from disk.
        
        # Used for change detection on the next sync 
        # If snapshot doesn't exist or is corrupted, returns empty dict (triggers full reindex).

        snapshot_path = Path(self.snapshot_file)
        
        if not snapshot_path.exists():
            return {}
        
        try:
            with open(snapshot_path, "r") as f:
                return json.load(f)
        except Exception:
            # If snapshot is corrupted, return empty (next sync will trigger full reindex)
            return {}

    def _save_snapshot(self, metadata: Dict[str, Any]) -> None:
        
        # Save the current metadata snapshot to disk.
        
        snapshot_path = Path(self.snapshot_file)
        
        try:
            with open(snapshot_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            # Log error but don't fail the sync
            print(f"Warning: Could not save snapshot: {e}")

    def force_full_reindex(self) -> Dict[str, Any]:
        
        return self.sync(force_full_reindex=True)


class SyncScheduler:
    # Scheduler for periodic schema synchronization.
    
    # Used to periodically check for schema changes and update the vector index in the background
    

    def __init__(self, synchronizer: SchemaSynchronizer) -> None:
        self.synchronizer = synchronizer
        self._running = False

    def sync_once(self) -> Dict[str, Any]:
        return self.synchronizer.sync()

    def start_periodic_sync(self, interval_seconds: int = 3600) -> None:
        try:
            from threading import Thread
            import time
        except ImportError:
            print("Threading not available; periodic sync disabled.")
            return
        
        def _sync_loop():
            while self._running:
                try:
                    result = self.synchronizer.sync()
                    if result["success"]:
                        added = result.get("tables_added", 0)
                        removed = result.get("tables_removed", 0)
                        modified = result.get("tables_modified", 0)
                        unchanged = result.get("tables_unchanged", 0)
                        
                        if added > 0 or removed > 0 or modified > 0:
                            print(f"[Schema Sync] Changes detected: "
                                  f"+{added} -{removed} ~{modified} "
                                  f"(unchanged: {unchanged})")
                except Exception as e:
                    print(f"[Schema Sync] Error: {e}")
                
                time.sleep(interval_seconds)
        
        self._running = True
        thread = Thread(target=_sync_loop, daemon=True)
        thread.start()

    def stop_periodic_sync(self) -> None:
        self._running = False
