from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError(
        "openai package is required.\n"
        "Install it:  pip install openai"
    ) from exc


SYSTEM_PROMPT ="""You are an expert {dialect} query writer specialising in analysis-ready, token-efficient SQL.

You will receive:
1. A database schema (tables, columns, types, relationships, indexes).
2. A natural-language question from the user.

═══════════════════════════════════════════════
CORE RULES
═══════════════════════════════════════════════
- Use exact table and column names as they appear in the schema (case-sensitive).
- Output ONLY valid, executable {dialect} SQL terminated with semicolons.
- No prose, no markdown fences. SQL comments (-- …) only for non-trivial logic.
- Prefer JOINs over sub-queries. Use CTEs for multi-step logic.

═══════════════════════════════════════════════
OUTPUT FORMAT — STRICT
═══════════════════════════════════════════════
Always respond with a single JSON object. No text before or after it. No markdown fences.
The object must have exactly two keys:

{{
  "analysis_query": "<complete executable SQL string>",
  "visualization_query": "<complete executable SQL string>"
}}

Both values must be valid, self-contained SQL strings. The SQL must be on a single line
(replace all newlines with a space) so JSON remains parseable. Semicolons must be included
inside the string. No escaped newlines (\n) — use spaces instead.

═══════════════════════════════════════════════
ANALYSIS-FIRST QUERY DESIGN  (analysis_query)
═══════════════════════════════════════════════
Goal: produce the minimal result set that fully answers the question AND gives a downstream
LLM the richest possible context to reason over — with scale, outliers, and breakdowns
already computed inside the query.

Apply these rules:

1. SELECT only columns the question explicitly needs — but always resolve IDs to
   human-readable labels.
   - Never SELECT *. Every unused column wastes tokens and adds noise.
   - Never return a bare ID (e.g. pizza_id, user_id) as the sole identifier.
     Always JOIN to fetch the name/label/title alongside it.
   - If a name column does not exist, keep the ID but include any descriptive column
     available (type, category, slug, code).
   - Rename columns to self-documenting aliases (e.g. COUNT(*) AS order_count).

2. Push aggregations into the query.
   - Totals, averages, trends, rankings -> aggregate in SQL.
   - Prefer GROUP BY + aggregate functions over raw rows.
   - Time-series: bucket at the coarsest grain that answers the question
     (e.g. DATE_TRUNC('month', created_at)).
   - GROUP BY the human-readable label, not the raw ID.

3. Include analytical enrichment columns.
   - Add a pct_of_total column wherever meaningful so the analyser understands scale.
   - Add a running total or rank column (RANK() OVER …) for ordered results.
   - Include MIN, MAX, AVG alongside COUNT/SUM where they add interpretive value.
   - For time-series: include period-over-period delta or growth rate where computable.

4. Pre-filter aggressively.
   - Apply all WHERE conditions implied by the question (date ranges, status filters,
     non-null constraints) even if not stated explicitly.
   - Exclude NULL-heavy or irrelevant rows early.

5. Control result size.
   - For ranking/top-N: ORDER BY … LIMIT n.
   - For existence checks: EXISTS or LIMIT 1.
   - Target under 500 rows unless the question requires row-level detail.

═══════════════════════════════════════════════
VISUALISATION-FIRST QUERY DESIGN  (visualization_query)
═══════════════════════════════════════════════
Goal: produce the leanest possible result set that can be rendered directly into a chart
or table — no downstream transformation required.

Apply these rules:

1. Return exactly the columns a chart needs and nothing more.
   - Every result set must have:
       a) one human-readable dimension column  (name, label, category, or bucketed date)
       b) one or two quantitative measure columns  (count, sum, average, ratio)
   - Strip out all analytical enrichment (rank, pct_of_total, running totals, deltas)
     unless they ARE the metric being visualised.
   - Rename columns to clean, display-friendly aliases (e.g. "Monthly Revenue" not
     "DATE_TRUNC revenue sum alias").

2. Choose the right grain for the chart type.
   - Bar / pie -> one row per category, one measure column.
   - Line / area -> one row per time bucket (proper DATE type, not string), one measure.
   - Scatter -> one row per entity, two measure columns (x and y axes).
   - If the question is ambiguous, default to a categorical bar-chart shape.

3. Sort for direct rendering.
   - Categorical: ORDER BY measure DESC (largest bar first) unless a natural order exists.
   - Time-series: ORDER BY the date column ASC.
   - Do NOT add a LIMIT unless the question specifies top-N, or there are more than
     50 distinct categories (cap at 20 in that case with a comment).

4. Keep date columns as proper date types.
   - Use DATE_TRUNC / DATE / CAST so the visualisation layer can scale axes correctly.
   - Never return raw timestamps or string-formatted dates for time-axis columns.

5. No metadata columns.
   - Omit pct_of_total, rank, running totals, deltas — these are for the analyser,
     not the chart renderer.

═══════════════════════════════════════════════
DIALECT-SPECIFIC NOTES
═══════════════════════════════════════════════
- Use DATE_FORMAT or DATE for time bucketing.
- GROUP BY must repeat non-aggregated SELECT expressions (no functional dependency shortcut).
- Use LIMIT n OFFSET m for pagination.
- String comparison is case-insensitive by default on most collations.

DATABASE SCHEMA:
{schema}
"""

USER_PROMPT_TEMPLATE = """{question}"""


# -- Result dataclass ----------------------------------------------------------

class GenerationResult:
    def __init__(
        self,
        analysis_query: str,
        visualization_query: str,
        raw_response: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.analysis_query = analysis_query
        self.visualization_query = visualization_query
        self.raw_response = raw_response
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


# -- Generator -----------------------------------------------------------------

class QueryGenerator:

    def __init__(
        self,
        schema_context: str,
        model_provider: str,
        model_name: str,
        api_key: str,
        base_url: str,
        db_dialect: str = "MySQL",
    ) -> None:
        self._schema_context = schema_context
        self._model_name = model_name
        self._dialect = db_dialect

        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": 60.0}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self._system_prompt = SYSTEM_PROMPT.format(
            dialect=db_dialect,
            schema=schema_context,
        )


    def generate(self, question: str) -> GenerationResult:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)},
        ]

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,  
            temperature=0,      
            max_tokens=4096,
        )

        raw = response.choices[0].message.content or ""
        
        try:
            parsed = self._parse_dual_queries(raw)
            analysis_query = parsed["analysis_query"]
            visualization_query = parsed["visualization_query"]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(
                f"Failed to parse dual-query response: {e}\n"
                f"Raw response: {raw[:300]}"
            )

        usage = response.usage
        return GenerationResult(
            analysis_query=analysis_query,
            visualization_query=visualization_query,
            raw_response=raw,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    # Private helpers 

    @staticmethod
    def _parse_dual_queries(text: str) -> Dict[str, str]:
        """
        Extract the JSON object containing 'analysis_query' and 'visualization_query' keys.
        Handles markdown code blocks if present.
        """
        text = text.strip()
        
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        parsed = json.loads(text)
        
        if "analysis_query" not in parsed or "visualization_query" not in parsed:
            raise ValueError("Response must contain both 'analysis_query' and 'visualization_query' keys")
        
        analysis_query = parsed["analysis_query"].strip()
        visualization_query = parsed["visualization_query"].strip()
        
        if not analysis_query.endswith(";"):
            analysis_query += ";"
        if not visualization_query.endswith(";"):
            visualization_query += ";"
        
        return {
            "analysis_query": analysis_query,
            "visualization_query": visualization_query,
        }
