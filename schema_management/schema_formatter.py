
from __future__ import annotations

from typing import Any, Dict, List

# Converts raw schema metadata into a compact, LLM-friendly text representation
class SchemaFormatter:

    def __init__(self, schema_data: Dict[str, Any]) -> None:
        self._data = schema_data

    def format(self) -> str:
        parts: List[str] = []

        parts.append(
            f"DATABASE: {self._data['target_database']}  "
            f"(MySQL {self._data.get('server_version', '?')})\n"
        )

        for schema, tables in self._data["tables"].items():
            for tbl in tables:
                parts.append(self._format_table(schema, tbl))

        if self._data["views"]:
            parts.append("VIEWS:")
            for v in self._data["views"]:
                parts.append(f"  {v['schema']}.{v['name']}")
            parts.append("")

        if self._data["foreign_keys"]:
            parts.append("RELATIONSHIPS (Foreign Keys):")
            for fk in self._data["foreign_keys"]:
                cascade = ""
                if fk["on_delete"] not in ("", "NO ACTION", "RESTRICT"):
                    cascade = f"  ON DELETE {fk['on_delete']}"
                if fk["on_update"] not in ("", "NO ACTION", "RESTRICT"):
                    cascade += f"  ON UPDATE {fk['on_update']}"
                parts.append(
                    f"  {fk['from_table']}.{fk['from_column']}"
                    f" -> {fk['to_table']}.{fk['to_column']}"
                    f"{cascade}"
                )
            parts.append("")

        return "\n".join(parts)
    
    # Private: table formatting

    def _format_table(self, schema: str, tbl: Dict[str, Any]) -> str:
        table_key = f"{schema}.{tbl['name']}"
        columns = self._data["columns"].get(table_key, [])
        pks = set(self._data["primary_keys"].get(table_key, []))
        indexes = self._data["indexes"].get(table_key, [])

        lines: List[str] = [f"TABLE: {table_key}"]
        if tbl["comment"]:
            lines.append(f"  -- {tbl['comment']}")

        lines.append("  COLUMNS:")
        for col in columns:
            annotations: List[str] = []
            if col["name"] in pks:
                annotations.append("PK")
            if "auto_increment" in (col["extra"] or "").lower():
                annotations.append("AUTO_INCREMENT")
            if col["key"] == "UNI":
                annotations.append("UNIQUE")
            if not col["nullable"]:
                nullability = "NOT NULL"
            else:
                nullability = "NULL"
            default_str = ""
            if col["default"] is not None:
                default_str = f"  DEFAULT {col['default']}"
            ann_str = f"  [{', '.join(annotations)}]" if annotations else ""
            comment_str = f"  -- {col['comment']}" if col["comment"] else ""
            col_type = (col["type"] or col["data_type"] or "?").upper()
            lines.append(
                f"    {col['name']:<30} {col_type:<20} {nullability}"
                f"{default_str}{ann_str}{comment_str}"
            )

        if indexes:
            idx_map: Dict[str, Dict] = {}
            for idx in indexes:
                nm = idx["name"]
                if nm not in idx_map:
                    idx_map[nm] = {
                        "unique": idx["unique"],
                        "type": idx["type"],
                        "columns": [],
                    }
                idx_map[nm]["columns"].append(idx["column"])
            lines.append("  INDEXES:")
            for nm, meta in idx_map.items():
                unique_str = "unique, " if meta["unique"] else ""
                lines.append(
                    f"    {nm} ({unique_str}{meta['type']}) "
                    f"-> {', '.join(meta['columns'])}"
                )

        if tbl["approx_rows"] is not None:
            rows_fmt = f"{int(tbl['approx_rows']):,}"
            lines.append(f"  APPROX ROWS: {rows_fmt}")

        lines.append("") 
        return "\n".join(lines)
