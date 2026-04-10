from __future__ import annotations

import json
import time
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
from openai import OpenAI

# Holds insights and visualization code from the open source model (Mistral 7B)
class AnalysisResult:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        column_names: List[str],
        inferred_dtypes: Dict[str, str],
        summary_stats: Dict[str, Any],
        record_count: int,
        insights: str,
        visualization_code: str,
        total_tokens: int,
        elapsed_time: float = 0.0,
    ) -> None:
        self.dataframe = dataframe
        self.column_names = column_names
        self.inferred_dtypes = inferred_dtypes
        self.summary_stats = summary_stats
        self.record_count = record_count
        self.insights = insights
        self.visualization_code = visualization_code
        self.total_tokens = total_tokens
        self.elapsed_time = elapsed_time


# Converts SQL results to DataFrame and generates insights + viz code via Mistral
class AnalysisGenerator:

    def __init__(self, api_key: str, base_url: str, model_name: str = "mistral",) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model_name = model_name
        
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=30.0,  
        )

    # Convert SQL result rows to DataFrame, prepare context, and generate insights + visualization.
    def analyze(self, rows: List[Dict[str, Any]], max_records: int = 100, stream_callback: Optional[Callable[[str], None]] = None,) -> AnalysisResult:
        
        start_time = time.time()
        
        df = pd.DataFrame(rows)
        
        column_names = list(df.columns)
        record_count = len(df)
        
        inferred_dtypes: Dict[str, str] = {}
        for col in df.columns:
            inferred_dtypes[col] = str(df[col].dtype)
        
        summary_stats = df.describe(include="all").to_dict()
        
        records_for_context = df.head(max_records).to_dict(orient="records")

        categorical_columns = {
           col: {
               "unique_count":     int(df[col].nunique()),
               "max_label_length": int(df[col].astype(str).str.len().max()),
           }
           for col in df.select_dtypes(exclude="number").columns
       }

        numeric_columns = list(df.select_dtypes(include="number").columns)
        
        enriched_context = {
            "column_names": column_names,
            "inferred_dtypes": inferred_dtypes,
            "record_count": record_count,
            "summary_statistics": summary_stats,
            "sample_records": records_for_context,
            "max_records_shown": len(records_for_context),
            "categorical_columns": categorical_columns,
            "numeric_columns": numeric_columns,
        }
        
        context_json = json.dumps(enriched_context, indent=2, default=str)
        
        analysis_response = self._generate_analysis_and_visualization(
            context_json,
            stream_callback=stream_callback,
        )
        
        insights = analysis_response["insights"]
        visualization_code = analysis_response["visualization"]
        total_tokens = analysis_response["tokens_used"]
        
        elapsed_time = time.time() - start_time
        
        return AnalysisResult(
            dataframe=df,
            column_names=column_names,
            inferred_dtypes=inferred_dtypes,
            summary_stats=summary_stats,
            record_count=record_count,
            insights=insights,
            visualization_code=visualization_code,
            total_tokens=total_tokens,
            elapsed_time=elapsed_time,
        )


    def _generate_analysis_and_visualization(self,enriched_context: str,stream_callback: Optional[Callable[[str], None]] = None,) -> Dict[str, Any]:
        
        system_prompt = '''MASTER CLASSIFICATION — DO THIS FIRST

VISUALIZATION NOT NEEDED (set to null) if ANY apply:
✗ Single value or single row result
✗ Simple list without numeric measurements
✗ Fewer than 3 data points
✗ All values nearly identical — no visual contrast
✗ Text conveys the information more clearly than a chart

VISUALIZATION NEEDED if ALL apply:
✓ At least 3 comparable data points
✓ At least one numeric column with meaningful values
✓ A visual pattern/trend/distribution provides clear insight

CHART TYPE OVERRIDE — APPLY BEFORE CHART TYPE SELECTION

If chart would be "bar", check categorical_columns[x_axis.column] from the data:
  → unique_count > 8 OR max_label_length > 8: MUST use "barh". No exceptions.
Additional rules when using "barh":
→ If label_count > 15, set top_n = 15
→ If max label length > 20, set label_truncate_length = 20
→ figure_height = max(6, label_count × 0.45)
→ rotation_x_labels = 0

CHART TYPE SELECTION

BARH → category vs numeric, > 8 categories OR labels > 8 chars (see override above)
BAR → category vs numeric, ≤ 8 categories, labels ≤ 8 chars
LINE     → trend over ordered/time x-axis
SCATTER  → numeric vs numeric correlation
HISTOGRAM→ single numeric distribution (10+ values)
PIE      → part-to-whole, 3–7 segments only
BOX      → spread/outliers across groups
HEATMAP  → 2D numeric intensity matrix
AREA     → cumulative trend over time

OUTPUT JSON STRUCTURE

Visualization needed:
{
  "insights": [{"insight": "..."}],
  "visualization": {
    "chart_type": "barh|bar|line|scatter|histogram|pie|box|heatmap|area",
    "title": "Clear, descriptive title",
    "top_n": null,
    "label_truncate_length": null,
    "x_axis": {"column": "exact column name", "label": "Human-readable label"},
    "y_axis": {"column": "exact column name", "label": "Human-readable label"},
    "styling": {
      "color_palette": "viridis",
      "alpha": 0.8,
      "grid": true,
      "rotation_x_labels": 0,
      "figure_width": 12,
      "figure_height": 6
    }
  }
}

Visualization not needed:
{"insights": [{"insight": "..."}], "visualization": null}

VALIDATION RULES

1. x_axis.column and y_axis.column MUST exactly match column names in the data
2. Data types:
   - BAR/BARH/PIE/BOX: categorical x (or y for barh), numeric metric
   - LINE/AREA: ordered/date x, numeric y
   - SCATTER: both axes numeric
   - HISTOGRAM: numeric y only
3. Mixed numeric/non-numeric columns: drop non-numeric rows
4. All-NaN column: pick a different column or return null
5. Timestamps: format as readable dates

INSIGHTS

- Max 5 insights, 1–2 sentences each
- Every insight MUST cite specific numbers from the data
- No invented benchmarks or external assumptions
- Focus on patterns, gaps, and anomalies

CRITICAL: Output valid JSON only — balanced braces/quotes, no trailing commas,
no NaN/Infinity/undefined. top_n and label_truncate_length may be null.'''

        user_message = f"""Analyze this dataset:

{enriched_context}

CRITICAL: Respond with ONLY a raw JSON object. No prose, no explanation, no markdown fences. 
Start your response with {{ and end with }}. Any text outside the JSON will break the parser."""

        prompt_tokens = 0
        completion_tokens = 0
        full_response = ""
        
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=4096, 
                stream=False, 
            )
            
            try:
                full_response = response.choices[0].message.content
                if full_response is None:
                    full_response = ""
            except (AttributeError, IndexError, TypeError) as e:
                raise RuntimeError(f"Failed to extract response from Ollama: {e}")
            
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            
        except TimeoutError as e:
            raise RuntimeError(
                f"Ollama request timed out after 60 seconds.\n"
                f"Check that:\n"
                f"  1. Ollama is running: ollama list\n"
                f"  2. Model '{self._model_name}' is installed: ollama pull {self._model_name}\n"
                f"  3. Ollama is accessible at: {self._base_url}\n"
                f"Original error: {e}"
            )
        except ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}\n"
                f"Make sure Ollama is running (visit https://ollama.ai to download)\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")
        
        cleaned_json = self._clean_json(full_response)
        
        try:
            parsed_result = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            repaired_json = self._repair_truncated_json(cleaned_json)
            try:
                parsed_result = json.loads(repaired_json)
            except json.JSONDecodeError as e2:
                raise ValueError(
                    f"Failed to parse unified JSON response: {e}\n"
                    f"Raw response ({len(cleaned_json)} chars): {cleaned_json[:300]}\n"
                    f"Repair attempt also failed: {e2}\n"
                )
        
        if "insights" not in parsed_result or "visualization" not in parsed_result:
            raise ValueError("Response must contain both 'insights' and 'visualization' keys")
        
        insights_str = json.dumps(parsed_result["insights"], default=str)
        
        if parsed_result["visualization"] is None:
            visualization_str = "null"
        else:
            visualization_str = json.dumps(parsed_result["visualization"], default=str)
        
        return {
            "insights": insights_str,
            "visualization": visualization_str,
            "tokens_used": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def _clean_json(text: str) -> str:

        text = text.strip()
        
        if text.startswith("```json"):
            text = text[7:]  
        elif text.startswith("```"):
            text = text[3:]  
        
        if text.endswith("```"):
            text = text[:-3]
        
        return text.strip()
    
    @staticmethod
    def _repair_truncated_json(text: str) -> str:

        text = text.strip()
        
        unescaped_quotes = 0
        i = 0
        while i < len(text):
            if text[i] == '"' and (i == 0 or text[i-1] != '\\'):
                unescaped_quotes += 1
            i += 1
        
        
        if unescaped_quotes % 2 == 1:
            text = text.rstrip()
            while text and text[-1] in ', \t\n':
                text = text[:-1]
            text += '"'  
        
        text = text.rstrip()
        if text.endswith(','):
            text = text[:-1]
        
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        text += ']' * open_brackets
        text += '}' * open_braces
        
        return text
