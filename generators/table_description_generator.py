from __future__ import annotations

from typing import Any, Dict, List, Optional
from openai import OpenAI


class TableDescriptionGenerator:
    # Generates intelligent table and column descriptions using Mistral.
    

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:11434/v1",
        model_name: str = "mistral",
    ) -> None:

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_table_descriptions(
        self,
        table_name: str,
        columns_info: List[Dict[str, Any]],
        pk_info: List[str],
        fk_info: List[Dict[str, Any]],
        sample_rows: List[Dict[str, Any]] = None,
        existing_comment: str = None,
    ) -> Dict[str, str]:
        
        # Build context about the table
        schema_context = self._build_schema_context(
            table_name, columns_info, pk_info, fk_info, sample_rows, existing_comment
        )
        
        # Generate table description
        table_description = self._generate_description(
            context=schema_context,
            description_type="table",
        )
        
        # Generate column descriptions
        column_descriptions = {}
        for col in columns_info:
            col_context = self._build_column_context(
                col, table_name, pk_info, fk_info, columns_info
            )
            col_desc = self._generate_description(
                context=col_context,
                description_type="column",
            )
            column_descriptions[col["name"]] = col_desc
        
        return {
            "table_description": table_description,
            "column_descriptions": column_descriptions,
        }

    def _build_schema_context(
        self,
        table_name: str,
        columns_info: List[Dict[str, Any]],
        pk_info: List[str],
        fk_info: List[Dict[str, Any]],
        sample_rows: Optional[List[Dict[str, Any]]],
        existing_comment: Optional[str],
    ) -> str:
        lines = []
        
        lines.append(f"Table Name: {table_name}")
        if existing_comment:
            lines.append(f"Existing Comment: {existing_comment}")
        
        lines.append("\nColumns:")
        for col in columns_info:
            col_line = f"  - {col['name']} ({col['type']})"
            flags = []
            if col["name"] in pk_info:
                flags.append("PRIMARY KEY")
            if not col.get("nullable", True):
                flags.append("NOT NULL")
            if col.get("comment"):
                flags.append(f"comment: {col['comment']}")
            if flags:
                col_line += f" [{', '.join(flags)}]"
            lines.append(col_line)
        
        if pk_info:
            lines.append(f"\nPrimary Keys: {', '.join(pk_info)}")
        
        if fk_info:
            lines.append("\nForeign Key Relationships:")
            for fk in fk_info:
                from_col = fk.get("from_column", "?")
                to_table = f"{fk.get('to_schema', '')}.{fk.get('to_table', '')}"
                to_col = fk.get("to_column", "?")
                direction = "references" if fk.get("from_table") == table_name else "referenced by"
                lines.append(f"  - {from_col} {direction} {to_table}.{to_col}")
        
        if sample_rows:
            lines.append(f"\nSample Data ({len(sample_rows)} rows):")
            for i, row in enumerate(sample_rows[:2], 1): 
                lines.append(f"  Row {i}: {row}")
        
        return "\n".join(lines)

    def _build_column_context(
        self,
        column: Dict[str, Any],
        table_name: str,
        pk_info: List[str],
        fk_info: List[Dict[str, Any]],
        all_columns: List[Dict[str, Any]],
    ) -> str:
        lines = []
        
        lines.append(f"Column: {column['name']}")
        lines.append(f"Type: {column['type']}")
        lines.append(f"Nullable: {column.get('nullable', True)}")
        
        if column["name"] in pk_info:
            lines.append("Is Primary Key: Yes")
        
        for fk in fk_info:
            if fk.get("from_column") == column["name"] and fk.get("from_table") == table_name:
                lines.append(f"Foreign Key: References {fk.get('to_schema')}.{fk.get('to_table')}.{fk.get('to_column')}")
        
        if column.get("comment"):
            lines.append(f"Existing Comment: {column['comment']}")
        
        return "\n".join(lines)

    def _generate_description(
        self,
        context: str,
        description_type: str = "table",
    ) -> str:

        if description_type == "table":
            prompt = self._build_table_prompt(context)
        else:
            prompt = self._build_column_prompt(context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Return a basic description if generation fails
            return f"[Description generation failed: {str(e)}]"

    def _build_table_prompt(self, context: str) -> str:
        return f"""Analyze the following database table schema and generate a comprehensive description.

{context}

Please provide:
1. A 2-3 sentence description of what this table represents and its purpose
2. Key business meaning and use cases
3. What can be derived from this table combined with its foreign key relationships (e.g., if there's an order table with order_id FK to customers table, you could derive customer-specific insights)
4. Any important patterns or constraints to note

Be concise but informative. Focus on what a SQL analyst or data scientist would need to know about this table."""

    def _build_column_prompt(self, context: str) -> str:
        return f"""Analyze the following database column and generate a brief, clear description.

{context}

Please provide:
1. A 1-2 sentence description of what this column contains and represents
2. Business meaning and how it's typically used
3. Any important patterns (e.g., if it's a status field, mention likely values; if it's a timestamp, mention its purpose)

Be concise and specific. Keep it under 50 words."""

    def generate_batch_descriptions(
        self,
        tables: List[Dict[str, Any]],
        schema_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, str]]:

        target_schema = schema_data.get("target_database", "")
        results = {}
        
        for table in tables:
            table_name = table["name"]
            full_table_name = f"{target_schema}.{table_name}"
            
            columns_info = schema_data["columns"].get(full_table_name, [])
            pk_info = schema_data["primary_keys"].get(full_table_name, [])
            fk_info = self._get_table_foreign_keys(full_table_name, schema_data)
            existing_comment = table.get("comment", "")
            
            descriptions = self.generate_table_descriptions(
                table_name=table_name,
                columns_info=columns_info,
                pk_info=pk_info,
                fk_info=fk_info,
                sample_rows=None,
                existing_comment=existing_comment,
            )
            
            results[full_table_name] = descriptions
        
        return results

    def _get_table_foreign_keys(
        self,
        full_table_name: str,
        schema_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        fks = []
        for fk in schema_data.get("foreign_keys", []):
            from_table = f"{fk['from_schema']}.{fk['from_table']}"
            to_table = f"{fk['to_schema']}.{fk['to_table']}"
            
            if from_table == full_table_name or to_table == full_table_name:
                fk_with_context = dict(fk)
                fk_with_context["from_table"] = fk["from_table"]  
                fks.append(fk_with_context)
        
        return fks
