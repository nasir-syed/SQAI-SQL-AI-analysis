from __future__ import annotations

import re
import socket
from typing import Any, Dict, List, Optional, Tuple
import mysql.connector
from mysql.connector import Error as MySQLError

def _validate_host(host: str) -> str:
    host = host.strip()
    if not host:
        raise ValueError("Host cannot be empty.")
    if not re.match(r'^[\w.\-]+$', host):
        raise ValueError(f"Invalid host: '{host}'")
    return host


def _validate_port(port: Any) -> int:
    try:
        p = int(port)
    except (TypeError, ValueError):
        raise ValueError(f"Port must be an integer, got: '{port}'")
    if not (1 <= p <= 65535):
        raise ValueError(f"Port must be 1–65535, got: {p}")
    return p


def _validate_database(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Database name cannot be empty.")
    if not re.match(r'^[\w\-$]+$', name):
        raise ValueError(
            f"Database name contains invalid characters: '{name}'. "
            "Only letters, digits, underscores, hyphens, and $ are allowed."
        )
    return name

# Manages a MySQL connection lifecycle
class DatabaseConnector:

    def __init__(self, credentials: Dict[str, Any]) -> None:
        self.host: str = _validate_host(credentials["host"])
        self.port: int = _validate_port(credentials.get("port", 3306))
        self.user: str = credentials["username"].strip()
        self.password: str = credentials.get("password", "")
        self.database: str = _validate_database(credentials["database"])
        self._conn: Optional[mysql.connector.MySQLConnection] = None


    def connect(self) -> None:
        self._check_host_reachable()
        try:
            self._conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                connection_timeout=10,
                charset="utf8mb4",
            )
            if not self._conn.is_connected():
                raise ConnectionError("Connection returned but is not active.")
        except MySQLError as exc:
            raise ConnectionError(f"MySQL error [{exc.errno}]: {exc.msg}") from exc

    def disconnect(self) -> None:
        if self._conn and self._conn.is_connected():
            self._conn.close()

    def execute_query(
        self,
        sql: str,
        params: Optional[Tuple] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_connected()
        cursor = self._conn.cursor(dictionary=True)
        try:
            cursor.execute(sql, params)
            return cursor.fetchall()
        except MySQLError as exc:
            raise RuntimeError(f"Query failed [{exc.errno}]: {exc.msg}\nSQL: {sql}") from exc
        finally:
            cursor.close()

    def server_version(self) -> str:
        self._ensure_connected()
        rows = self.execute_query("SELECT VERSION() AS v")
        return rows[0]["v"] if rows else "unknown"

    @property
    def connection(self) -> mysql.connector.MySQLConnection:
        self._ensure_connected()
        return self._conn 
 
    # Private helpers

    def _ensure_connected(self) -> None:
        if not self._conn or not self._conn.is_connected():
            raise ConnectionError(
                "Not connected to the database. Call connect() first."
            )

    def _check_host_reachable(self) -> None: 
        try:
            with socket.create_connection((self.host, self.port), timeout=5):
                pass
        except OSError as exc:
            raise ConnectionError(
                f"Cannot reach {self.host}:{self.port} — {exc}"
            ) from exc
