from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pandas as pd

from db_connector import DatabaseConnector
from generators.query_generator import QueryGenerator
from generators.analysis_generator import AnalysisGenerator
from schema_management.schema_retrieval import SchemaRetriever

# Main orchestration class for the advanced text-to-SQL pipeline
class AdvancedQueryPipeline:

    def __init__(
        self,
        db_connector: DatabaseConnector,
        query_generator: QueryGenerator,
        analysis_generator: AnalysisGenerator,
        schema_retriever: Optional[SchemaRetriever],
        model_provider: str,
        api_key: str,
        base_url: str,
        db_dialect: str = "MySQL",
        schema_context: str = "",
    ) -> None:
       
        self.db = db_connector
        self.query_gen = query_generator
        self.analysis_gen = analysis_generator
        self.schema_retriever = schema_retriever
        self.schema_context = schema_context
        
        self.model_provider = model_provider
        self.api_key = api_key
        self.base_url = base_url
        self.db_dialect = db_dialect

    def run(self, user_query: str) -> Dict[str, Any]:
        
        # Execute the full pipeline for a user query.
        
        # Returns a dict with:
        #   - user_query: Original query
        #   - retrieved_tables: List of table names retrieved via vector search
        #   - enriched_schema_context: Schema context used for SQL generation
        #   - sql_queries: {"analysis_query": ..., "visualization_query": ...}
        #   - execution_results: SQL execution results (DataFrame)
        #   - analysis: Text analysis from Mistral
        #   - visualization_code: Python code for visualization (via matplotlib)
        #   - timing: Timing information for each stage
        
        timing = {}
        result = {
            "user_query": user_query,
            "retrieved_tables": [],
            "enriched_schema_context": "",
            "sql_queries": {},
            "execution_results": None,
            "analysis": "",
            "visualization_code": "",
            "timing": timing,
        }
        
        start = time.time()
        enriched_context = ""
        
        if self.schema_retriever is not None:
            try:
                retrieved_docs = self.schema_retriever.retrieve(user_query)
                timing["schema_retrieval"] = time.time() - start
                result["retrieved_tables"] = [
                    doc["metadata"].get("full_table_name", "?") for doc in retrieved_docs
                ]
            except Exception as e:
                result["error"] = f"Schema retrieval failed: {str(e)}"
                timing["schema_retrieval"] = time.time() - start
                return result
            
            if not retrieved_docs:
                result["error"] = "No relevant tables found in vector store. Run schema sync first."
                return result
        
            start = time.time()
            enriched_context = self.schema_retriever.format_for_llm(
                user_query, 
                retrieved_docs
            )
            timing["context_formatting"] = time.time() - start
            result["enriched_schema_context"] = enriched_context
        else:        
            timing["schema_retrieval"] = time.time() - start
            result["retrieved_tables"] = ["*all_tables*"]
            enriched_context = self.schema_context
            result["enriched_schema_context"] = enriched_context
         
        start = time.time()
        try:
            sql_result = self._generate_sql_with_context(
                user_query,
                enriched_context
            )
            result["sql_queries"] = sql_result
            timing["sql_generation"] = time.time() - start
        except Exception as e:
            result["error"] = f"SQL generation failed: {str(e)}"
            timing["sql_generation"] = time.time() - start
            return result
        
        start = time.time()
        try:
            exec_results = self._execute_queries(sql_result)
            result["execution_results"] = exec_results
            timing["sql_execution"] = time.time() - start
        except Exception as e:
            result["error"] = f"SQL execution failed: {str(e)}"
            timing["sql_execution"] = time.time() - start
            return result
        
        start = time.time()
        try:
            analysis_rows = []
            analysis_df = exec_results.get("analysis_df")
            if analysis_df is not None and not analysis_df.empty:
                analysis_rows = analysis_df.to_dict(orient="records")
            
            if analysis_rows:
                analysis_result = self.analysis_gen.analyze(
                    rows=analysis_rows,
                    max_records=100,
                )
                result["analysis"] = analysis_result.insights
                result["visualization_code"] = analysis_result.visualization_code
            else:
                result["analysis"] = "No analysis data available"
                result["visualization_code"] = ""
            
            timing["analysis_generation"] = time.time() - start
        except Exception as e:
            error_msg = f"Analysis generation failed: {str(e)}"
            result["analysis_error"] = error_msg
            timing["analysis_generation"] = time.time() - start
        
        return result

    def _generate_sql_with_context(
        self,
        user_query: str,
        enriched_context: str,
    ) -> Dict[str, str]:
        
        temp_generator = QueryGenerator(
            schema_context=enriched_context,
            model_provider=self.model_provider,
            model_name=self.query_gen._model_name,  
            api_key=self.api_key,
            base_url=self.base_url,
            db_dialect=self.db_dialect,
        )

        result_obj = temp_generator.generate(user_query)
        
        return {
            "analysis_query": result_obj.analysis_query,
            "visualization_query": result_obj.visualization_query,
        }

    def _execute_queries(
        self,
        sql_result: Dict[str, str],
    ) -> Dict[str, Any]:

        result = {}
        
        if "analysis_query" in sql_result and sql_result["analysis_query"]:
            try:
                analysis_rows = self.db.execute_query(sql_result["analysis_query"])
                result["analysis_df"] = pd.DataFrame(analysis_rows)
            except Exception as e:
                result["analysis_error"] = str(e)
        
        if "visualization_query" in sql_result and sql_result["visualization_query"]:
            try:
                viz_rows = self.db.execute_query(sql_result["visualization_query"])
                result["visualization_df"] = pd.DataFrame(viz_rows)
            except Exception as e:
                result["visualization_error"] = str(e)
        
        return result