from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from db_connector import DatabaseConnector
from generators.table_description_generator import TableDescriptionGenerator

# Generates structured, embeddable documents from database schema metadata
class SchemaDocumentGenerator:

    def __init__(
        self,
        schema_data: Dict[str, Any],
        db_connector: DatabaseConnector,
        samples_per_table: int = 5,
        description_generator: Optional[TableDescriptionGenerator] = None,
    ) -> None:

        self.schema_data = schema_data
        self.db = db_connector
        self.samples_per_table = samples_per_table
        self.description_gen = description_generator

    def generate_all_documents(self) -> Dict[str, Dict[str, Any]]:
        
        # Generate documents for all tables in the schema
        
        documents = {}
        target_schema = self.schema_data["target_database"]
        
        tables = self.schema_data["tables"].get(target_schema, [])
        
        for table in tables:
            table_name = table["name"]
            full_table_name = f"{target_schema}.{table_name}"
            
            doc = self.generate_table_document(table_name, target_schema)
            documents[full_table_name] = doc
        
        return documents

    def generate_table_document(self, table_name: str, schema_name: str) -> Dict[str, Any]:
        
        # Generate a single table document.
        
        full_table_name = f"{schema_name}.{table_name}"
        
        # Get table metadata
        table_info = self._get_table_info(table_name, schema_name)
        columns_info = self._get_columns_info(full_table_name)
        pk_info = self._get_primary_keys_info(full_table_name)
        fk_info = self._get_foreign_keys_info(full_table_name)
        idx_info = self._get_indexes_info(full_table_name)
        sample_rows = self._get_sample_rows(full_table_name)
        
        # Generate descriptions 
        descriptions = None
        if self.description_gen:
            try:
                descriptions = self.description_gen.generate_table_descriptions(
                    table_name=table_name,
                    columns_info=columns_info,
                    pk_info=pk_info,
                    fk_info=fk_info,
                    sample_rows=sample_rows,
                    existing_comment=table_info.get("comment", ""),
                )
            except Exception as e:
                print(f"Warning: Failed to generate descriptions for {full_table_name}: {e}")
                descriptions = None
        
        # Format content for embedding
        content = self._format_document_content(
            table_info=table_info,
            columns_info=columns_info,
            pk_info=pk_info,
            fk_info=fk_info,
            idx_info=idx_info,
            sample_rows=sample_rows,
            descriptions=descriptions,
        )
        
        # Create metadata
        metadata = {
            "table_name": table_name,
            "schema_name": schema_name,
            "full_table_name": full_table_name,
            "row_count": table_info.get("approx_rows", 0),
            "engine": table_info.get("engine", ""),
            "generated_at": datetime.utcnow().isoformat(),
            "column_count": len(columns_info),
        }
        
        return {
            "content": content,
            "metadata": metadata,
            "table_name": full_table_name,
        }

    # Private methods 
    def _get_table_info(self, table_name: str, schema_name: str) -> Dict[str, Any]:
        tables = self.schema_data["tables"].get(schema_name, [])
        for table in tables:
            if table["name"] == table_name:
                return table
        return {}

    def _get_columns_info(self, full_table_name: str) -> List[Dict[str, Any]]:
        return self.schema_data["columns"].get(full_table_name, [])

    def _get_primary_keys_info(self, full_table_name: str) -> List[str]:
        return self.schema_data["primary_keys"].get(full_table_name, [])

    def _get_foreign_keys_info(self, full_table_name: str) -> List[Dict[str, Any]]:
        fks = []
        for fk in self.schema_data.get("foreign_keys", []):
            from_table = f"{fk['from_schema']}.{fk['from_table']}"
            to_table = f"{fk['to_schema']}.{fk['to_table']}"
            
            if from_table == full_table_name or to_table == full_table_name:
                fks.append(fk)
        
        return fks

    def _get_indexes_info(self, full_table_name: str) -> List[Dict[str, Any]]:
        return self.schema_data.get("indexes", {}).get(full_table_name, [])

    def _get_sample_rows(self, full_table_name: str) -> List[Dict[str, Any]]:
        try:
            query = f"SELECT * FROM {full_table_name} LIMIT {self.samples_per_table}"
            rows = self.db.execute_query(query)
            return rows
        except Exception:
            return []

    def _format_document_content(
        self,
        table_info: Dict[str, Any],
        columns_info: List[Dict[str, Any]],
        pk_info: List[str],
        fk_info: List[Dict[str, Any]],
        idx_info: List[Dict[str, Any]],
        sample_rows: List[Dict[str, Any]],
        descriptions: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines = []
        
        table_name = table_info.get("name", "UNKNOWN")
        comment = table_info.get("comment", "")
        row_count = table_info.get("approx_rows", 0)
        
        lines.append(f"TABLE: {table_name}")
        
        if descriptions and descriptions.get("table_description"):
            lines.append("\nTABLE DESCRIPTION:")
            lines.append(descriptions["table_description"])
        elif comment:
            lines.append(f"Description: {comment}")
        
        lines.append(f"\nApproximate Rows: {row_count:,}")
        lines.append("")
        
        lines.append("COLUMNS:")
        col_descriptions = descriptions.get("column_descriptions", {}) if descriptions else {}
        
        for col in columns_info:
            col_def = f"  - {col['name']}: {col['type']}"
            
            flags = []
            if col["name"] in pk_info:
                flags.append("PRIMARY KEY")
            if not col.get("nullable", True):
                flags.append("NOT NULL")
            if col.get("key") == "UNI":
                flags.append("UNIQUE")
            if col.get("extra"):
                flags.append(col["extra"].upper())
            
            if flags:
                col_def += f" ({', '.join(flags)})"
            
            if col["name"] in col_descriptions:
                col_def += f"\n      Description: {col_descriptions[col['name']]}"
            elif col.get("comment"):
                col_def += f"\n      Description: {col['comment']}"
            
            lines.append(col_def)
        
        lines.append("")
        
        if pk_info:
            lines.append("PRIMARY KEYS:")
            for pk in pk_info:
                lines.append(f"  - {pk}")
            lines.append("")
        
        if fk_info:
            lines.append("FOREIGN KEYS:")
            for fk in fk_info:
                from_col = fk.get("from_column", "?")
                to_table = f"{fk.get('to_schema', '')}.{fk.get('to_table', '')}"
                to_col = fk.get("to_column", "?")
                lines.append(f"  - {from_col} -> {to_table}.{to_col}")
            lines.append("")
        
        if idx_info:
            lines.append("INDEXES:")
            for idx in idx_info:
                idx_name = idx.get("name", "UNKNOWN")
                idx_cols = ", ".join(idx.get("columns", []))
                lines.append(f"  - {idx_name}: ({idx_cols})")
            lines.append("")
        
        if sample_rows:
            lines.append(f"SAMPLE DATA ({len(sample_rows)} rows):")
            df = pd.DataFrame(sample_rows)
            if len(df.columns) > 8:
                df = df.iloc[:, :8] 
            
            sample_str = df.to_string(index=False, max_rows=self.samples_per_table)
            for line in sample_str.split("\n"):
                lines.append(f"  {line}")
        
        return "\n".join(lines)

    def get_document_metadata_snapshot(self) -> Dict[str, Any]:
        snapshot = {}
        target_schema = self.schema_data["target_database"]
        tables = self.schema_data["tables"].get(target_schema, [])
        
        for table in tables:
            table_name = table["name"]
            full_table_name = f"{target_schema}.{table_name}"
            
            columns = self._get_columns_info(full_table_name)
            col_defs = [
                {
                    "name": col["name"],
                    "type": col["type"],
                    "nullable": col.get("nullable", True),
                }
                for col in columns
            ]
            
            snapshot[full_table_name] = {
                "columns": col_defs,
                "primary_keys": self._get_primary_keys_info(full_table_name),
                "foreign_keys": [
                    {
                        "from_column": fk.get("from_column"),
                        "to_table": f"{fk.get('to_schema')}.{fk.get('to_table')}",
                        "to_column": fk.get("to_column"),
                    }
                    for fk in self._get_foreign_keys_info(full_table_name)
                ],
                "row_count": table.get("approx_rows", 0),
                "comment": table.get("comment", ""),
            }
        
        return snapshot
