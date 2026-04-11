from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from generators.query_generator import GenerationResult

# ANSI colour helpers

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_MAGENTA= "\033[35m"
_WHITE  = "\033[97m"


def _c(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


class CLI:

    def print_banner(self) -> None:
        banner = r"""
  ____   ___    _    ___ 
 / ___| / _ \  / \  |_ _|
 \___ \| | | |/ _ \  | | 
  ___) | |_| / ___ \ | | 
 |____/ \__\_/_/   \_\___|  LLM-Powered SQL Analysis
"""
        print(_c(banner, _CYAN, _BOLD))
        print(_c(" " + "-" * 40, _DIM))
        print()


    def print_step(self, number: int, message: str) -> None:
        print(_c(f"\n[{number}] {message}", _BLUE, _BOLD))

    def print_success(self, message: str) -> None:
        print(_c(f"    [OK] {message}", _GREEN))

    def print_error(self, message: str) -> None:
        print(_c(f"\n  [ERROR] {message}\n", _RED, _BOLD), file=sys.stderr)

    def print_warning(self, message: str) -> None:
        print(_c(f"  [!] {message}", _YELLOW))

    def print_info(self, message: str) -> None:
        print(_c(f"  --> {message}", _DIM))

    def prompt_query(self) -> str:
        print()
        try:
            raw = input(_c("  Ask > ", _MAGENTA, _BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "exit"
        return raw

    def print_schema_preview(self, schema_context: str) -> None:
        lines = schema_context.splitlines()
        preview_lines = lines[:60]
        truncated = len(lines) > 60

        print()
        print(_c("  -- Schema Preview " + "-" * 22, _DIM))
        for line in preview_lines:
            print(_c("  " + line, _DIM))
        if truncated:
            print(_c(f"  … ({len(lines) - 60} more lines)", _DIM))
        print(_c("  " + "-" * 40, _DIM))

    def print_sql_result(self, result: GenerationResult) -> None:
        print()
        print(
            _c(f"  -- Generated SQL ", _GREEN, _BOLD) +
            _c(
                f"(model: {result.model}  "
                f"tokens in/out: {result.input_tokens}/{result.output_tokens})",
                _DIM,
            )
        )
        print(_c("  " + "-" * 60, _DIM))

        if len(result.statements) == 1:
            self._print_sql_block(result.sql)
        else:
            for i, stmt in enumerate(result.statements, 1):
                print(_c(f"\n  -- Statement {i} of {len(result.statements)}", _DIM))
                self._print_sql_block(stmt + ";")

        print(_c("  " + "-" * 60, _DIM))
        print()

    def execute_and_display(self, result: GenerationResult, connector: Any) -> tuple:
        print()
        print(
            _c(f"  -- Generated SQL ", _GREEN, _BOLD) +
            _c(
                f"(model: {result.model}  "
                f"tokens in/out: {result.input_tokens}/{result.output_tokens})",
                _DIM,
            )
        )
        print(_c("  " + "-" * 60, _DIM))

        if len(result.statements) == 1:
            self._print_sql_block(result.sql)
        else:
            for i, stmt in enumerate(result.statements, 1):
                print(_c(f"\n  -- Statement {i} of {len(result.statements)}", _DIM))
                self._print_sql_block(stmt + ";")

        print(_c("  " + "-" * 60, _DIM))

        validation_error = self._validate_sql(result.sql)
        if validation_error:
            print(_c(f"\n  ✗ SQL Validation Error: {validation_error}", _RED))
            print()
            return [], True

        print(_c("\n  -- Execution Results ", _BLUE, _BOLD) + _c("-" * 40, _DIM))
        try:
            rows = connector.execute_query(result.sql)
            if not rows:
                print(_c("  (No rows returned)\n", _DIM))
                return [], False
            else:
                self._display_results_table(rows)
                print()
                return rows, False
        except Exception as exc:
            print(_c(f"\n  ✗ Execution Error: {exc}\n", _RED))
            return [], True

    @staticmethod
    def _validate_sql(sql: str) -> Optional[str]:
        sql_clean = sql.strip()
        
        if not sql_clean:
            return "Empty SQL query"
        
        sql_upper = sql_clean.upper()
        
        valid_starts = ('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH')
        if not any(sql_upper.startswith(cmd) for cmd in valid_starts):
            return f"Query must start with one of: {', '.join(valid_starts)}"
        
        if sql_clean.count('(') != sql_clean.count(')'):
            return "Unmatched parentheses in SQL query"
        
        single_quotes = [i for i, c in enumerate(sql_clean) if c == "'" and (i == 0 or sql_clean[i-1] != '\\')]
        if len(single_quotes) % 2 != 0:
            return "Unmatched single quotes in SQL query"
        
        return None 

    @staticmethod
    def _display_results_table(rows: List[Dict[str, Any]]) -> None:
        # Format and display query results as a table
        if not rows:
            return

        columns = list(rows[0].keys())
        
        col_widths = {col: max(len(str(col)), max(len(str(row.get(col, ''))) for row in rows)) for col in columns}
        
        header = "  " + " | ".join(col.ljust(col_widths[col]) for col in columns)
        print(_c(header, _CYAN, _BOLD))
        print(_c("  " + "".join("-" * (col_widths[col] + 3) for col in columns), _DIM))
        
        for row in rows:
            row_str = "  " + " | ".join(
                str(row.get(col, '')).ljust(col_widths[col]) for col in columns
            )
            print(row_str)

    def display_analysis(self, analysis: Any) -> None:
        if not analysis:
            return

        print(_c("\n  -- Analysis Metadata ", _MAGENTA, _BOLD) + _c("-" * 38, _DIM))
        print(_c(f"    Records: {analysis.record_count}", _WHITE))
        print(_c(f"    Columns: {len(analysis.column_names)}", _WHITE))
        print(_c(f"    ({', '.join(analysis.column_names[:5])}{'...' if len(analysis.column_names) > 5 else ''})", _DIM))
        print(_c(f"    Elapsed time: {analysis.elapsed_time:.2f}s", _YELLOW))
        print(_c(f"    Total tokens used: {analysis.total_tokens}", _DIM))
        print()
        
        print(_c("  -- Inferred Data Types ", _MAGENTA, _BOLD) + _c("-" * 36, _DIM))
        for col, dtype in analysis.inferred_dtypes.items():
            print(_c(f"    {col}: {dtype}", _WHITE))
        print()
        
        print(_c("  -- Summary Statistics (via df.describe) ", _MAGENTA, _BOLD) + _c("-" * 22, _DIM))
        if analysis.summary_stats:
            for stat_name, stat_values in analysis.summary_stats.items():
                print(_c(f"    {stat_name}:", _CYAN))
                for col, val in list(stat_values.items())[:10]:
                    print(_c(f"      {col}: {val}", _DIM))
            print()
        
        print(_c("  -- Analytical Insights (via Mistral) ", _GREEN, _BOLD) + _c("-" * 25, _DIM))
        print()
        
        import json
        try:
            insights_array = json.loads(analysis.insights)
            if isinstance(insights_array, list):
                for i, item in enumerate(insights_array, 1):
                    if isinstance(item, dict) and "insight" in item:
                        print(_c(f"    {i}. {item['insight']}", _WHITE))
                    else:
                        print(_c(f"    {i}. {str(item)}", _WHITE))
            else:
                print(_c(f"    {analysis.insights}", _WHITE))
        except (json.JSONDecodeError, ValueError):
            print(_c(f"    {analysis.insights}", _WHITE))
        print()
        
        print(_c("  -- Visualization Specification (JSON) ", _BLUE, _BOLD) + _c("-" * 24, _DIM))
        print()
        print(_c("  " + "-" * 60, _DIM))
         
        try:
            viz_spec = json.loads(analysis.visualization_code)
            viz_json_str = json.dumps(viz_spec, indent=2)
            for line in viz_json_str.split('\n'):
                wrapped_lines = self._wrap_line(line, max_width=75)
                for wrapped in wrapped_lines:
                    print(_c(f"    {wrapped}", _CYAN))
        except (json.JSONDecodeError, ValueError):
            code_lines = analysis.visualization_code.split('\n')
            for line in code_lines:
                wrapped_lines = self._wrap_line(line, max_width=75)
                for wrapped in wrapped_lines:
                    print(_c(f"    {wrapped}", _CYAN))


    # Private helpers

    @staticmethod
    def _wrap_line(line: str, max_width: int = 75) -> List[str]:
        # Wrap a line to fit terminal width
        if len(line) <= max_width:
            return [line]
        
        wrapped = []
        current = line
        while len(current) > max_width:
            wrapped.append(current[:max_width])
            current = "  " + current[max_width:] 
        if current:
            wrapped.append(current)
        return wrapped

    def _print_sql_block(self, sql: str) -> None:
        # Syntax-highlight SQL keywords for terminals 
        keywords = {
            "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER",
            "OUTER", "ON", "GROUP", "BY", "ORDER", "HAVING", "LIMIT",
            "OFFSET", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
            "DELETE", "CREATE", "TABLE", "DROP", "ALTER", "INDEX",
            "AND", "OR", "NOT", "NULL", "IS", "IN", "LIKE", "BETWEEN",
            "AS", "DISTINCT", "COUNT", "SUM", "AVG", "MAX", "MIN",
            "CASE", "WHEN", "THEN", "ELSE", "END", "UNION", "ALL",
            "WITH", "EXISTS", "COALESCE", "IFNULL", "IF",
        }

        for line in sql.splitlines():
            highlighted = line
            if sys.stdout.isatty():
                for kw in keywords:
                    import re
                    highlighted = re.sub(
                        rf'\b({re.escape(kw)})\b',
                        _c(r'\1', _CYAN, _BOLD),
                        highlighted,
                        flags=re.IGNORECASE,
                    )
                highlighted = re.sub(
                    r"('(?:[^']|'')*')",
                    _c(r'\1', _YELLOW),
                    highlighted,
                )
                highlighted = re.sub(
                    r'(--.*)',
                    _c(r'\1', _DIM),
                    highlighted,
                )
            print("  " + highlighted) 

    def print_section_analysis_results(self, analysis_text: Optional[str], tokens: int) -> None:
        # Display analytical insights from Mistral
        if not analysis_text:
            return
        
        print()
        print(_c("  -- Analytical Insights ", _GREEN, _BOLD) + 
              _c(f"({tokens} tokens)", _DIM))
        print(_c("  " + "-" * 60, _DIM))
        print()
        for line in analysis_text.split('\n'):
            if line.strip():
                print(_c(f"    {line}", _WHITE))
        print()

    def print_section_visualization_code(self, viz_code: Optional[str], tokens: int) -> None:
        # Display generated visualization code
        if not viz_code:
            return
        
        print()
        print(_c("  -- Visualization Code (Python)", _MAGENTA, _BOLD) + 
              _c(f"({tokens} tokens)", _DIM))
        print(_c("  " + "-" * 60, _DIM))
        print()
        
        for line in viz_code.split('\n')[:50]:
            if line.strip():
                print(_c(f"    {line}", _YELLOW))
        
        if len(viz_code.split('\n')) > 50:
            print(_c(f"    ... ({len(viz_code.split(chr(10))) - 50} more lines)", _DIM))
        
        print()
