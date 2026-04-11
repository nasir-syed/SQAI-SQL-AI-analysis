import sys
import json
from typing import Optional, Dict, Any
from pathlib import Path

# Adjust path to import parent modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from db_connector import DatabaseConnector
from schema_management.schema_inspector import SchemaInspector
from schema_management.schema_formatter import SchemaFormatter
from schema_management.schema_document_generator import SchemaDocumentGenerator
from generators.table_description_generator import TableDescriptionGenerator
from vector_store import VectorStoreManager
from schema_management.schema_sync import SchemaSynchronizer, SyncScheduler
from schema_management.schema_retrieval import SchemaRetriever
from generators.query_generator import QueryGenerator
from generators.analysis_generator import AnalysisGenerator
from query_pipeline import AdvancedQueryPipeline


def _check_ollama_running(base_url: str) -> bool:
    try:
        import requests
        health_url = base_url.replace("/v1", "") + "/api/tags"
        response = requests.get(health_url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


# Manages the entire analysis pipeline
class PipelineManager:

    def __init__(self):
        self.config = Config()
        self.connector: Optional[DatabaseConnector] = None
        self.pipeline: Optional[AdvancedQueryPipeline] = None
        self.schema_context: Optional[str] = None
        self._initialized = False
        self._init_error: Optional[str] = None

    def initialize(self) -> bool:
        try:
            #Connect to database
            credentials = self.config.get_db_credentials()
            self.connector = DatabaseConnector(credentials)
            self.connector.connect()

            #Introspect schema
            inspector = SchemaInspector(self.connector)
            schema_data = inspector.extract()

            #Vector store + schema sync
            use_vector_retrieval = False
            schema_retriever = None

            if self.config.ENABLE_VECTOR_RETRIEVAL:
                try:
                    #Initialize vector store
                    vector_mgr = VectorStoreManager(
                        persist_dir=self.config.VECTOR_DB_PATH,
                    )

                    # nitialize table description generator
                    description_gen = None
                    if _check_ollama_running(self.config.OLLAMA_BASE_URL):
                        try:
                            description_gen = TableDescriptionGenerator(
                                api_key=self.config.OLLAMA_API_KEY,
                                base_url=self.config.OLLAMA_BASE_URL,
                                model_name=self.config.OLLAMA_MODEL,
                            )
                        except Exception:
                            description_gen = None

                    #Build document generator
                    doc_generator = SchemaDocumentGenerator(
                        schema_data=schema_data,
                        db_connector=self.connector,
                        samples_per_table=self.config.SAMPLES_PER_TABLE,
                        description_generator=description_gen,
                    )

                    #Synchronize schema with vector DB
                    synchronizer = SchemaSynchronizer(
                        schema_inspector=inspector,
                        doc_generator=doc_generator,
                        vector_store_manager=vector_mgr,
                        snapshot_file=self.config.SCHEMA_SNAPSHOT_PATH,
                    )

                    sync_result = synchronizer.sync(
                        force_full_reindex=self.config.FORCE_FULL_REINDEX_ON_STARTUP
                    )

                    if not sync_result["success"]:
                        raise RuntimeError(
                            f"Schema sync failed: {sync_result.get('error', 'Unknown error')}"
                        )

                    if self.config.ENABLE_PERIODIC_SYNC:
                        scheduler = SyncScheduler(synchronizer)
                        scheduler.start_periodic_sync(
                            interval_seconds=self.config.SCHEMA_SYNC_INTERVAL
                        )

                    #Initialize schema retriever
                    schema_retriever = SchemaRetriever(
                        vector_store_manager=vector_mgr,
                        default_top_k=self.config.VECTOR_TOP_K,
                    )
                    use_vector_retrieval = True

                except Exception as e:
                    self._init_error = f"Vector DB initialization failed: {e}"
                    use_vector_retrieval = False

            #Format schema as LLM context
            formatter = SchemaFormatter(schema_data)
            self.schema_context = formatter.format()

            #Initialize query generator
            query_gen = QueryGenerator(
                schema_context=self.schema_context,
                model_provider=self.config.MODEL_PROVIDER,
                model_name=self.config.MODEL_NAME,
                api_key=self.config.API_KEY,
                base_url=self.config.BASE_URL,
                db_dialect=credentials.get("dialect", "MySQL"),
            )

            #Initialize analysis generator
            if not _check_ollama_running(self.config.OLLAMA_BASE_URL):
                raise RuntimeError(
                    f"Ollama is not running or not reachable at {self.config.OLLAMA_BASE_URL}. "
                    f"Start Ollama and pull the model: ollama pull {self.config.OLLAMA_MODEL}"
                )   

            analysis_gen = AnalysisGenerator(
                api_key=self.config.OLLAMA_API_KEY,
                base_url=self.config.OLLAMA_BASE_URL,
                model_name=self.config.OLLAMA_MODEL,
            )

            #Require vector retrieval
            if not use_vector_retrieval:
                raise RuntimeError(
                    self._init_error or "Vector retrieval is required but could not be initialized."
                )

            # The full pipeline
            self.pipeline = AdvancedQueryPipeline(
                db_connector=self.connector,
                query_generator=query_gen,
                analysis_generator=analysis_gen,
                schema_retriever=schema_retriever,
                model_provider=self.config.MODEL_PROVIDER,
                api_key=self.config.API_KEY,
                base_url=self.config.BASE_URL,
                db_dialect=credentials.get("dialect", "MySQL"),
                schema_context=self.schema_context,
            )

            self._initialized = True
            return True

        except Exception as e:
            self._init_error = str(e)
            if self.connector:
                try:
                    self.connector.disconnect()
                except Exception:
                    pass
            return False

    # Takes the user query and runs it through the pipeline
    def execute_query(self, user_query: str) -> Dict[str, Any]:
        if not self._initialized or not self.pipeline:
            return {
                "error": "Pipeline not initialized. " + (self._init_error or "Unknown error"),
                "success": False,
            }

        try:
            result = self.pipeline.run(user_query)
            result["success"] = "error" not in result
            return result
        except Exception as e:
            return {
                "error": f"Query execution failed: {str(e)}",
                "success": False,
            }

    def cleanup(self) -> None:
        if self.connector:
            try:
                self.connector.disconnect()
            except Exception:
                pass

#Parses and validates visualization specifications from Mistral
class VisualizationParser:

    @staticmethod
    def parse(viz_json_str: str) -> Optional[Dict[str, Any]]:
        if not viz_json_str or viz_json_str.strip() == "":
            return None

        if viz_json_str.strip().lower() == "null":
            return None

        try:
            viz_spec = json.loads(viz_json_str)
            if viz_spec is None:
                return None
            
            if isinstance(viz_spec, dict):
                if "visualization" in viz_spec:
                    return viz_spec.get("visualization")
                elif "chart_type" in viz_spec:
                    return viz_spec
                elif len(viz_spec) > 0:
                    return viz_spec
            
            return None
        except json.JSONDecodeError as e:
            import sys
            print(f"DEBUG: Failed to parse visualization JSON: {str(e)}", file=sys.stderr)
            print(f"DEBUG: Input was: {viz_json_str[:200]}", file=sys.stderr)
            return None

    @staticmethod
    def validate(spec: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        required_fields = ["chart_type", "title"]

        for field in required_fields:
            if field not in spec:
                return False, f"Missing required field: {field}"

        valid_chart_types = [
            "bar", "barh", "line", "scatter", "histogram", "box",
            "heatmap", "pie", "area"
        ]

        if spec["chart_type"] not in valid_chart_types:
            return False, f"Invalid chart_type: {spec['chart_type']}"

        return True, None

    @staticmethod
    def get_chart_type(spec: Dict[str, Any]) -> str:
        return spec.get("chart_type", "bar")

    @staticmethod
    def get_title(spec: Dict[str, Any]) -> str:
        return spec.get("title", "Chart")

    @staticmethod
    def get_x_axis(spec: Dict[str, Any]) -> Dict[str, Any]:
        return spec.get("x_axis", {})

    @staticmethod
    def get_y_axis(spec: Dict[str, Any]) -> Dict[str, Any]:
        return spec.get("y_axis", {})

    @staticmethod
    def get_styling(spec: Dict[str, Any]) -> Dict[str, Any]:
        return spec.get("styling", {})

    @staticmethod
    def get_grouping(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        grouping = spec.get("grouping", {})
        return grouping if grouping.get("enabled") else None

    @staticmethod
    def get_aggregation(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        agg = spec.get("aggregation", {})
        return agg if agg.get("enabled") else None