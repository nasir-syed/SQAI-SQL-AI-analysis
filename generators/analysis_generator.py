from __future__ import annotations

import json
import time
import re
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
    def analyze(self, rows: List[Dict[str, Any]], max_records: int = 250, stream_callback: Optional[Callable[[str], None]] = None,) -> AnalysisResult:
        
        start_time = time.time()

        clean_rows = []
        for row in rows:
            clean_row = {}
            for k, v in row.items():
                if hasattr(v, '__class__') and v.__class__.__name__ == 'Decimal':
                    clean_row[k] = float(v)
                else:
                    clean_row[k] = v
            clean_rows.append(clean_row)
        
        df = pd.DataFrame(clean_rows)
        
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
        
        insights = '[]'
        insights_tokens = 0
        try:
            insights_response = self._generate_insights(context_json, stream_callback=stream_callback)
            insights = insights_response["insights"]
            insights_tokens = insights_response["tokens_used"]
        except Exception as e:
            default_insights = [{"insight": f"Unable to generate insights: {str(e)}"}]
            insights = json.dumps(default_insights, default=str)
            insights_tokens = 0
        
        visualization_code = "null"
        visualization_tokens = 0
        try:
            visualization_response = self._generate_visualization(context_json, stream_callback=stream_callback)
            visualization_code = visualization_response["visualization"]
            visualization_tokens = visualization_response["tokens_used"]
        except Exception as e:
            visualization_code = "null"
            visualization_tokens = 0
        
        total_tokens = insights_tokens + visualization_tokens
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



    def _generate_insights(self, enriched_context: str, stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        
        system_prompt = '''You are a data analyst. Analyze the dataset and generate 3-5 key insights.

RULES FOR INSIGHTS:
- BEFORE writing any insight, scan ALL rows to find true min/max values
- Every insight MUST cite specific numbers pulled directly from the data
- Superlatives (highest, lowest, most, least) MUST reflect the true extreme after scanning every row
- Max 5 insights, 1–2 sentences each
- No invented benchmarks or external assumptions
- Focus on patterns, gaps, and anomalies

RULES FOR SINGULAR/FEW VALUES:
- If a column contains only one or very few distinct values, state it directly and move on — no analysis needed
- Be punctual and direct: only mention the value and what it relates to (e.g. "Total revenue is $83,000.")
- Do NOT add context, interpretation, comparisons, patterns, or qualifiers like "for this dataset", "notably", "interestingly", etc.
- Do NOT reference mean, median, distribution, or any statistical framing — just the value and its label
- CRITICAL: Do NOT mention minimum, maximum, mean, 25th percentile, median, and 75th percentile for single / few values

OUTPUT JSON STRUCTURE:
Return ONLY valid JSON with no extra text or formatting:
{"insights": [{"insight": "First insight with specific numbers"}, {"insight": "Second insight"}]}

CRITICAL REQUIREMENTS:
- Every comma, colon, and quote MUST be properly placed
- NO line breaks inside the JSON object
- NO extra whitespace
- Start response with { and end with }
- ONLY the JSON, nothing else

CRITICAL: Respond with ONLY a raw JSON object. No prose, no explanation, no markdown fences.'''

        user_message = f"""Analyze this dataset and generate insights:

{enriched_context}

CRITICAL: Respond with ONLY the JSON object. Start with {{ and end with }}. No other text."""

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
                max_tokens=2048,
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
                f"Check that Ollama is running and model '{self._model_name}' is installed"
            )
        except ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}\n"
                f"Make sure Ollama is running"
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
                # If repair still fails, try the aggressive repair
                aggressive_json = self._repair_aggressive_json(cleaned_json)
                try:
                    parsed_result = json.loads(aggressive_json)
                except json.JSONDecodeError as e3:
                    import sys
                    print(f"DEBUG: Failed to parse insights JSON after repairs", file=sys.stderr)
                    print(f"DEBUG: Original response length: {len(full_response)}", file=sys.stderr)
                    print(f"DEBUG: Error: {str(e3)}", file=sys.stderr)
                    raise ValueError(f"Failed to parse insights JSON: {str(e3)}")
        
        if "insights" not in parsed_result:
            raise ValueError("Response must contain 'insights' key")
        
        insights_str = json.dumps(parsed_result["insights"], default=str)
        
        return {
            "insights": insights_str,
            "tokens_used": prompt_tokens + completion_tokens,
        }

    def _generate_visualization(self, enriched_context: str, stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        
        system_prompt = '''You are a data visualization expert. Generate a visualization specification for the dataset.

VISUALIZATION CLASSIFICATION:

DO NOT VISUALIZE if ANY apply:
✗ Single value or single row result
✗ Simple list without numeric measurements
✗ Fewer than 3 data points
✗ All values nearly identical — no visual contrast
✗ Text conveys information more clearly than a chart

DO VISUALIZE if ALL apply:
✓ At least 3 comparable data points
✓ At least one numeric column with meaningful values
✓ A visual pattern/trend/distribution provides clear insight

CHART TYPE OVERRIDE:
If chart would be "bar", check categorical_columns[x_axis.column]:
  → unique_count > 8 OR max_label_length > 8: MUST use "barh"
  → If using barh and label_count > 32, set top_n = 32
  → If max label length > 20, set label_truncate_length = 20
  → figure_height = max(6, label_count × 0.45)

VALIDATION:
1. x_axis.column and y_axis.column MUST exactly match column names
2. Data types: BAR/BARH/PIE/BOX need categorical x, numeric y
3. All column references must exist in the data

CHART TYPE SELECTION:
- BARH → category vs numeric, > 8 categories OR labels > 8 chars
- BAR → category vs numeric, ≤ 8 categories, labels ≤ 8 chars
- LINE → trend over ordered/time x-axis
- SCATTER → numeric vs numeric correlation
- HISTOGRAM → single numeric distribution (10+ values)
- PIE → part-to-whole, 3–7 segments only
- BOX → spread/outliers across groups
- HEATMAP → 2D numeric intensity matrix
- AREA → cumulative trend over time

OUTPUT JSON STRUCTURE:
If visualization is needed, return this exact JSON (NO extra text, NO markdown, NO explanations):
{"visualization": {"chart_type": "bar", "title": "Title", "top_n": null, "label_truncate_length": null, "x_axis": {"column": "col_name", "label": "Label"}, "y_axis": {"column": "col_name", "label": "Label"}, "styling": {"color_palette": "viridis", "alpha": 0.8, "grid": true, "rotation_x_labels": 0, "figure_width": 10, "figure_height": 5}}}

If NO visualization needed, return:
{"visualization": null}

CRITICAL REQUIREMENTS:
- Every comma, colon, and quote MUST be exactly as shown above
- NO line breaks inside the JSON object
- NO extra whitespace or formatting
- Start response with { and end with }
- ONLY the JSON, nothing else'''

        user_message = f"""Generate a visualization specification for this dataset:

{enriched_context}

CRITICAL: Respond with ONLY the JSON object. Start with {{ and end with }}. No other text."""

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
                f"Check that Ollama is running and model '{self._model_name}' is installed"
            )
        except ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}\n"
                f"Make sure Ollama is running"
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
                aggressive_json = self._repair_aggressive_json(cleaned_json)
                try:
                    parsed_result = json.loads(aggressive_json)
                except json.JSONDecodeError as e3:
                    import sys
                    print(f"DEBUG: Failed to parse JSON after repairs", file=sys.stderr)
                    print(f"DEBUG: Original response length: {len(full_response)}", file=sys.stderr)
                    print(f"DEBUG: Cleaned JSON: {cleaned_json[:500]}", file=sys.stderr)
                    print(f"DEBUG: Error: {str(e3)}", file=sys.stderr)
                    raise ValueError(f"Failed to parse visualization JSON: {str(e3)}")
        
        visualization_spec = None
        if isinstance(parsed_result, dict):
            if "visualization" in parsed_result:
                visualization_spec = parsed_result.get("visualization")
            elif "chart_type" in parsed_result or len(parsed_result) == 0:
                visualization_spec = parsed_result
        
        if visualization_spec is None:
            visualization_str = "null"
        else:
            visualization_str = json.dumps(visualization_spec, default=str)
        
        return {
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

    @staticmethod
    def _repair_aggressive_json(text: str) -> str:
        """Aggressively repair common JSON errors"""
        text = text.strip()

        text = re.sub(r'([\}\]])\s*"', r'\1,"', text)
        text = re.sub(r'([\}\]])\s*\{', r'\1,{', text)
        text = re.sub(r'([\}\]])\s*\[', r'\1,[', text)

        text = re.sub(r'("\s*)\s+"', r'\1,"', text)

        text = re.sub(r'("\s*:\s*[^,\}]+?)\s+(")', r'\1,\2', text)

        text = re.sub(r',(\s*[\}\]])', r'\1', text)

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
