import sys
import time
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
from cli import CLI


def _check_ollama_running(base_url: str) -> bool:
    try:
        import requests
        health_url = base_url.replace("/v1", "") + "/api/tags"
        response = requests.get(health_url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def main():
    cli = CLI()
    cli.print_banner()

    config = Config()

    # Get database credentials from config
    try:
        credentials = config.get_db_credentials()
    except ValueError as exc:
        cli.print_error(str(exc))
        sys.exit(1)

    # Connect to database
    cli.print_step(1, "Connecting to database…")
    connector = DatabaseConnector(credentials)
    try:
        connector.connect()
        cli.print_success(
            f"Connected to {credentials['host']}:{credentials['port']} "
            f"-> database '{credentials['database']}'"
        )
    except Exception as exc:
        cli.print_error(f"Connection failed: {exc}")
        sys.exit(1)

    # Introspect schema 
    cli.print_step(2, "Introspecting schema via INFORMATION_SCHEMA…")
    inspector = SchemaInspector(connector)
    try:
        schema_data = inspector.extract()
        table_count = sum(len(t) for t in schema_data["tables"].values())
        cli.print_success(
            f"Found {len(schema_data['schemas'])} schema(s), "
            f"{table_count} table(s)"
        )
    except Exception as exc:
        cli.print_error(f"Schema introspection failed: {exc}")
        connector.disconnect()
        sys.exit(1)

    # Initialize Vector Database and Schema Synchronization
    if config.ENABLE_VECTOR_RETRIEVAL:
        cli.print_step(3, "Initializing vector database for schema retrieval…")
        try:
            vector_mgr = VectorStoreManager(
                persist_dir=config.VECTOR_DB_PATH,
            )
            cli.print_success(
                f"Vector store initialized at {config.VECTOR_DB_PATH}"
            )
            
            description_gen = True
            if _check_ollama_running(config.OLLAMA_BASE_URL):
                try:
                    description_gen = TableDescriptionGenerator(
                        api_key=config.OLLAMA_API_KEY,
                        base_url=config.OLLAMA_BASE_URL,
                        model_name=config.OLLAMA_MODEL,
                    )
                    cli.print_info(
                        f"Table descriptions enabled (using {config.OLLAMA_MODEL} via Ollama)"
                    )
                except Exception as e:
                    cli.print_warning(f"Table description generator failed to initialize: {e}")
                    description_gen = None
            else:
                cli.print_warning("Ollama not available — table descriptions will use database comments")
            
            doc_generator = SchemaDocumentGenerator(
                schema_data=schema_data,
                db_connector=connector,
                samples_per_table=config.SAMPLES_PER_TABLE,
                description_generator=description_gen,
            )
            
            synchronizer = SchemaSynchronizer(
                schema_inspector=inspector,
                doc_generator=doc_generator,
                vector_store_manager=vector_mgr,
                snapshot_file=config.SCHEMA_SNAPSHOT_PATH,
            )
            
            cli.print_step(3, "Synchronizing schema with vector database…")
            sync_result = synchronizer.sync(
                force_full_reindex=config.FORCE_FULL_REINDEX_ON_STARTUP
            )
            
            if sync_result["success"]:
                cli.print_success(
                    f"Schema synced: {sync_result['tables_added']} added, "
                    f"{sync_result['tables_removed']} removed, "
                    f"{sync_result['tables_modified']} modified, "
                    f"{sync_result['total_tables']} total"
                )
            else:
                cli.print_error(f"Schema sync failed: {sync_result.get('error', 'Unknown error')}")
                connector.disconnect()
                sys.exit(1)
            
            if config.ENABLE_PERIODIC_SYNC:
                scheduler = SyncScheduler(synchronizer)
                scheduler.start_periodic_sync(interval_seconds=config.SCHEMA_SYNC_INTERVAL)
                cli.print_info(
                    f"Periodic schema sync enabled (every {config.SCHEMA_SYNC_INTERVAL}s)"
                )
            
            schema_retriever = SchemaRetriever(
                vector_store_manager=vector_mgr,
                default_top_k=config.VECTOR_TOP_K,
            )
            use_vector_retrieval = True
            
        except Exception as exc:
            cli.print_error(f"Vector DB initialization failed: {exc}")
            cli.print_warning("Falling back to traditional pipeline…")
            use_vector_retrieval = False
    else:
        cli.print_warning("Vector retrieval disabled — using traditional pipeline")
        use_vector_retrieval = False

    formatter = SchemaFormatter(schema_data)
    schema_context = formatter.format()

    if config.SHOW_SCHEMA_PREVIEW:
        cli.print_schema_preview(schema_context)

    generator = QueryGenerator(
        schema_context=schema_context,
        model_provider=config.MODEL_PROVIDER,
        model_name=config.MODEL_NAME,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        db_dialect=credentials.get("dialect", "MySQL"),
    )

    cli.print_step(len([3, 4, 5]) + 1, "Initializing Mistral for analysis…")
    
    if not _check_ollama_running(config.OLLAMA_BASE_URL):
        cli.print_error(
            f"ERROR: Ollama is not running or not reachable at {config.OLLAMA_BASE_URL}\n\n"
            f"To fix this:\n"
            f"  1. Start Ollama (visit https://ollama.ai to download)\n"
            f"  2. Pull the Mistral model: ollama pull mistral\n"
            f"  3. Ensure Ollama is accessible at {config.OLLAMA_BASE_URL}\n"
        )
        sys.exit(1)
    
    try:
        analysis_generator = AnalysisGenerator(
            api_key=config.OLLAMA_API_KEY,
            base_url=config.OLLAMA_BASE_URL,
            model_name=config.OLLAMA_MODEL,
        )
        cli.print_success(
            f"Ollama is running with model '{config.OLLAMA_MODEL}' at {config.OLLAMA_BASE_URL}"
        )
    except Exception as exc:
        cli.print_error(
            f"ERROR: Failed to initialize Mistral/Ollama:\n{exc}\n"
            f"Make sure model '{config.OLLAMA_MODEL}' is installed:\n"
            f"  ollama pull {config.OLLAMA_MODEL}"
        )
        sys.exit(1)

    cli.print_step(5, "Initializing Advanced Text-to-SQL Pipeline…")
    try:
        if use_vector_retrieval:
            pipeline = AdvancedQueryPipeline(
                db_connector=connector,
                query_generator=generator,
                analysis_generator=analysis_generator,
                schema_retriever=schema_retriever,
                model_provider=config.MODEL_PROVIDER,
                api_key=config.API_KEY,
                base_url=config.BASE_URL,
                db_dialect=credentials.get("dialect", "MySQL"),
            )
            cli.print_success("Pipeline ready — vector-based schema retrieval enabled")
        else:
            cli.print_error("Vector retrieval not available")
            connector.disconnect()
            sys.exit(1)
    except Exception as exc:
        cli.print_error(f"Pipeline initialization failed: {exc}")
        connector.disconnect()
        sys.exit(1)

    cli.print_step(6, "Ready — enter natural language queries (type 'exit' to quit)\n")

    while True:
        user_query = cli.prompt_query()

        if user_query.lower() in {"exit", "quit", "q", "\\q"}:
            cli.print_info("Goodbye!")
            break

        if not user_query.strip():
            cli.print_warning("Empty input — please enter a question.")
            continue

        cli.print_step(6, "Executing advanced query pipeline…")
        pipeline_start = time.time()
        try:
            result = pipeline.run(user_query)
            pipeline_duration = time.time() - pipeline_start
        except Exception as exc:
            cli.print_error(f"Pipeline execution failed: {exc}")
            continue
        
        if result.get("error"):
            cli.print_error(f"Pipeline error: {result['error']}")
            continue
        
        if result.get("analysis_error"):
            cli.print_warning(f"Analysis stage warning: {result['analysis_error']}")
        
        if result.get("retrieved_tables"):
            cli.print_info("\n" + "-" * 62)
            cli.print_info("RETRIEVED TABLES:")
            for table in result["retrieved_tables"]:
                cli.print_info(f"  + {table}")
            cli.print_info("-" * 62)
         
        if result.get("sql_queries"):
            cli.print_info("\n" + "-" * 62)
            cli.print_info("GENERATED SQL QUERIES:")
            cli.print_info("-" * 62)
            
            analysis_q = result["sql_queries"].get("analysis_query", "")
            viz_q = result["sql_queries"].get("visualization_query", "")
            
            if analysis_q:
                cli.print_info("\n[ANALYSIS QUERY]:")
                for line in analysis_q.split("\n"):
                    cli.print_info(f"  {line}")
            
            if viz_q:
                cli.print_info("\n[VISUALIZATION QUERY]:")
                for line in viz_q.split("\n"):
                    cli.print_info(f"  {line}")
            
            cli.print_info("-" * 62)
        
        if result.get("execution_results"):
            exec_results = result["execution_results"]
            analysis_df = exec_results.get("analysis_df")
            viz_df = exec_results.get("visualization_df")
            
            if analysis_df is not None and not analysis_df.empty:
                cli.print_info("\n" + "-" * 62)
                cli.print_info("ANALYSIS QUERY RESULTS:")
                cli.print_info(f"  Rows: {len(analysis_df)}, Columns: {len(analysis_df.columns)}")
                cli.print_info("-" * 62)
            
            if viz_df is not None and not viz_df.empty:
                cli.print_info("\n" + "-" * 62)
                cli.print_info("VISUALIZATION QUERY RESULTS:")
                cli.print_info(f"  Rows: {len(viz_df)}, Columns: {len(viz_df.columns)}")
                cli.print_info("-" * 62)
        
        if result.get("analysis"):
            cli.print_section_analysis_results(result["analysis"], 0)
        
        if result.get("visualization_code"):
            cli.print_section_visualization_code(result["visualization_code"], 0)
        
        timing = result.get("timing", {})
        
        cli.print_info("\n" + "-" * 62)
        cli.print_info("PIPELINE BREAKDOWN:")
        cli.print_info(f"  Schema Retrieval:      {timing.get('schema_retrieval', 0):7.2f}s")
        cli.print_info(f"  Context Formatting:    {timing.get('context_formatting', 0):7.2f}s")
        cli.print_info(f"  SQL Generation:        {timing.get('sql_generation', 0):7.2f}s")
        cli.print_info(f"  SQL Execution:         {timing.get('sql_execution', 0):7.2f}s")
        cli.print_info(f"  Analysis Generation:   {timing.get('analysis_generation', 0):7.2f}s")
        cli.print_info("-" * 62)
        cli.print_info(f"  TOTAL:                 {pipeline_duration:7.2f}s")
        cli.print_info("-" * 62)

    connector.disconnect()


if __name__ == "__main__":
    main()
