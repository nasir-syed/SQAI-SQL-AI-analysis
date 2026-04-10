from __future__ import annotations

from typing import Any, Dict, List

from db_connector import DatabaseConnector

# Extracts full schema metadata from MySQL via INFORMATION_SCHEMA.
class SchemaInspector:

    def __init__(self, connector: DatabaseConnector) -> None:
        self._db = connector
        self._target_db = connector.database

    def extract(self) -> Dict[str, Any]:
        """Return a single dict with all schema metadata."""
        server_version = self._db.server_version()

        schemas = self._get_schemas()
        tables = self._get_tables()
        columns = self._get_columns()
        primary_keys = self._get_primary_keys()
        foreign_keys = self._get_foreign_keys()
        indexes = self._get_indexes()
        views = self._get_views()

        return {
            "server_version": server_version,
            "target_database": self._target_db,
            "schemas": schemas,
            "tables": tables,         
            "columns": columns,       
            "primary_keys": primary_keys,  
            "foreign_keys": foreign_keys,  
            "indexes": indexes,       
            "views": views,           
        }

    # Private: schema list 

    def _get_schemas(self) -> List[str]:
        rows = self._db.execute_query(
            """
            SELECT SCHEMA_NAME
            FROM   INFORMATION_SCHEMA.SCHEMATA
            WHERE  SCHEMA_NAME NOT IN (
                       'information_schema', 'mysql',
                       'performance_schema', 'sys'
                   )
            ORDER  BY SCHEMA_NAME
            """
        )
        return [r["SCHEMA_NAME"] for r in rows]

    # Private: tables

    def _get_tables(self) -> Dict[str, List[Dict[str, Any]]]:
        rows = self._db.execute_query(
            """
            SELECT TABLE_SCHEMA,
                   TABLE_NAME,
                   TABLE_TYPE,
                   ENGINE,
                   TABLE_ROWS,
                   TABLE_COMMENT,
                   CREATE_TIME,
                   TABLE_COLLATION
            FROM   INFORMATION_SCHEMA.TABLES
            WHERE  TABLE_SCHEMA = %s
              AND  TABLE_TYPE   = 'BASE TABLE'
            ORDER  BY TABLE_SCHEMA, TABLE_NAME
            """,
            (self._target_db,),
        )
        result: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            schema = r["TABLE_SCHEMA"]
            result.setdefault(schema, []).append(
                {
                    "name": r["TABLE_NAME"],
                    "engine": r["ENGINE"],
                    "approx_rows": r["TABLE_ROWS"],
                    "comment": r["TABLE_COMMENT"] or "",
                    "collation": r["TABLE_COLLATION"] or "",
                }
            )
        return result

    # Private: columns 

    def _get_columns(self) -> Dict[str, List[Dict[str, Any]]]:
        rows = self._db.execute_query(
            """
            SELECT TABLE_SCHEMA,
                   TABLE_NAME,
                   COLUMN_NAME,
                   ORDINAL_POSITION,
                   COLUMN_DEFAULT,
                   IS_NULLABLE,
                   DATA_TYPE,
                   CHARACTER_MAXIMUM_LENGTH,
                   NUMERIC_PRECISION,
                   NUMERIC_SCALE,
                   COLUMN_TYPE,
                   COLUMN_KEY,
                   EXTRA,
                   COLUMN_COMMENT
            FROM   INFORMATION_SCHEMA.COLUMNS
            WHERE  TABLE_SCHEMA = %s
            ORDER  BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
            """,
            (self._target_db,),
        )
        result: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            key = f"{r['TABLE_SCHEMA']}.{r['TABLE_NAME']}"
            result.setdefault(key, []).append(
                {
                    "name": r["COLUMN_NAME"],
                    "type": r["COLUMN_TYPE"],
                    "data_type": r["DATA_TYPE"],
                    "nullable": r["IS_NULLABLE"] == "YES",
                    "default": r["COLUMN_DEFAULT"],
                    "key": r["COLUMN_KEY"],       # PRI / UNI / MUL
                    "extra": r["EXTRA"],           # auto_increment etc.
                    "comment": r["COLUMN_COMMENT"] or "",
                    "max_length": r["CHARACTER_MAXIMUM_LENGTH"],
                    "num_precision": r["NUMERIC_PRECISION"],
                    "num_scale": r["NUMERIC_SCALE"],
                }
            )
        return result

    # Private: primary keys

    def _get_primary_keys(self) -> Dict[str, List[str]]:
        rows = self._db.execute_query(
            """
            SELECT TABLE_SCHEMA,
                   TABLE_NAME,
                   COLUMN_NAME,
                   ORDINAL_POSITION
            FROM   INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE  CONSTRAINT_NAME = 'PRIMARY'
              AND  TABLE_SCHEMA    = %s
            ORDER  BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
            """,
            (self._target_db,),
        )
        result: Dict[str, List[str]] = {}
        for r in rows:
            key = f"{r['TABLE_SCHEMA']}.{r['TABLE_NAME']}"
            result.setdefault(key, []).append(r["COLUMN_NAME"])
        return result

    # Private: foreign keys 

    def _get_foreign_keys(self) -> List[Dict[str, Any]]:
        rows = self._db.execute_query(
            """
            SELECT kcu.CONSTRAINT_NAME,
                   kcu.TABLE_SCHEMA,
                   kcu.TABLE_NAME,
                   kcu.COLUMN_NAME,
                   kcu.REFERENCED_TABLE_SCHEMA,
                   kcu.REFERENCED_TABLE_NAME,
                   kcu.REFERENCED_COLUMN_NAME,
                   rc.UPDATE_RULE,
                   rc.DELETE_RULE
            FROM   INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
            JOIN   INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                   ON  rc.CONSTRAINT_NAME   = kcu.CONSTRAINT_NAME
                   AND rc.CONSTRAINT_SCHEMA = kcu.TABLE_SCHEMA
            WHERE  kcu.TABLE_SCHEMA              = %s
              AND  kcu.REFERENCED_TABLE_NAME IS NOT NULL
            ORDER  BY kcu.TABLE_NAME, kcu.CONSTRAINT_NAME
            """,
            (self._target_db,),
        )
        return [
            {
                "constraint": r["CONSTRAINT_NAME"],
                "from_schema": r["TABLE_SCHEMA"],
                "from_table": r["TABLE_NAME"],
                "from_column": r["COLUMN_NAME"],
                "to_schema": r["REFERENCED_TABLE_SCHEMA"],
                "to_table": r["REFERENCED_TABLE_NAME"],
                "to_column": r["REFERENCED_COLUMN_NAME"],
                "on_update": r["UPDATE_RULE"],
                "on_delete": r["DELETE_RULE"],
            }
            for r in rows
        ]

    # Private: indexes

    def _get_indexes(self) -> Dict[str, List[Dict[str, Any]]]:
        rows = self._db.execute_query(
            """
            SELECT TABLE_SCHEMA,
                   TABLE_NAME,
                   INDEX_NAME,
                   COLUMN_NAME,
                   NON_UNIQUE,
                   SEQ_IN_INDEX,
                   INDEX_TYPE
            FROM   INFORMATION_SCHEMA.STATISTICS
            WHERE  TABLE_SCHEMA = %s
              AND  INDEX_NAME  != 'PRIMARY'
            ORDER  BY TABLE_SCHEMA, TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
            """,
            (self._target_db,),
        )
        result: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            key = f"{r['TABLE_SCHEMA']}.{r['TABLE_NAME']}"
            result.setdefault(key, []).append(
                {
                    "name": r["INDEX_NAME"],
                    "column": r["COLUMN_NAME"],
                    "unique": r["NON_UNIQUE"] == 0,
                    "seq": r["SEQ_IN_INDEX"],
                    "type": r["INDEX_TYPE"],
                }
            )
        return result

    # Private: views

    def _get_views(self) -> List[Dict[str, Any]]:
        rows = self._db.execute_query(
            """
            SELECT TABLE_SCHEMA,
                   TABLE_NAME,
                   VIEW_DEFINITION,
                   IS_UPDATABLE,
                   CHECK_OPTION
            FROM   INFORMATION_SCHEMA.VIEWS
            WHERE  TABLE_SCHEMA = %s
            ORDER  BY TABLE_NAME
            """,
            (self._target_db,),
        )
        return [
            {
                "schema": r["TABLE_SCHEMA"],
                "name": r["TABLE_NAME"],
                "definition": r["VIEW_DEFINITION"],
                "updatable": r["IS_UPDATABLE"],
                "check_option": r["CHECK_OPTION"],
            }
            for r in rows
        ]
