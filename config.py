import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    MODEL_PROVIDER: str = "openai"
    MODEL_NAME: str = "gpt-4o"

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")

    SHOW_SCHEMA_PREVIEW: bool = True
    MAX_DISPLAY_ROWS: int = 50

    _enable_analysis_env = os.getenv("ENABLE_ANALYSIS", "true").strip().lower()
    ENABLE_ANALYSIS: bool = _enable_analysis_env not in ("false", "0", "no")
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")    
    OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "not-needed")
    
    MAX_ANALYSIS_RECORDS: int = int(os.getenv("MAX_ANALYSIS_RECORDS", "250"))

    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "")

    @property
    def API_KEY(self) -> str:
        
        key = self.OPENAI_API_KEY
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Export it: export OPENAI_API_KEY='sk-...'"
            )
        return key

    @property
    def BASE_URL(self) -> str:
        return self.OPENAI_BASE_URL or "" 

    def get_db_credentials(self) -> dict:
        if not self.MYSQL_DATABASE:
            raise ValueError(
                "MYSQL_DATABASE is not set in .env file.\n"
                "Add to .env:\n"
                "  MYSQL_HOST=localhost\n"
                "  MYSQL_PORT=3306\n"
                "  MYSQL_USER=root\n"
                "  MYSQL_PASSWORD=your_password\n"
                "  MYSQL_DATABASE=your_database"
            )
        
        return {
            "host": self.MYSQL_HOST,
            "port": self.MYSQL_PORT,
            "username": self.MYSQL_USER,
            "password": self.MYSQL_PASSWORD,
            "database": self.MYSQL_DATABASE,
            "dialect": "MySQL",
        }

    ENABLE_VECTOR_RETRIEVAL: bool = (
        os.getenv("ENABLE_VECTOR_RETRIEVAL", "true").strip().lower() 
        not in ("false", "0", "no")
    )
    
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./schema_embeddings")
    
    VECTOR_COLLECTION_NAME: str = os.getenv("VECTOR_COLLECTION_NAME", "schema_tables")
    
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "3"))
    
    SAMPLES_PER_TABLE: int = int(os.getenv("SAMPLES_PER_TABLE", "5"))
    
    SCHEMA_SNAPSHOT_PATH: str = os.getenv(
        "SCHEMA_SNAPSHOT_PATH", 
        "./schema_metadata_snapshot.json"
    )
    
    ENABLE_PERIODIC_SYNC: bool = (
        os.getenv("ENABLE_PERIODIC_SYNC", "true").strip().lower() 
        not in ("false", "0", "no")
    )
    
    SCHEMA_SYNC_INTERVAL: int = int(
        os.getenv("SCHEMA_SYNC_INTERVAL", "3600") # Default: 1 hour
    )
    
    FORCE_FULL_REINDEX_ON_STARTUP: bool = (
        os.getenv("FORCE_FULL_REINDEX_ON_STARTUP", "false").strip().lower()
        not in ("false", "0", "no")
    )
