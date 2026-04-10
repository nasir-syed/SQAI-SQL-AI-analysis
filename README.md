# SQAI - SQL-Based AI Analysis Platform

## Overview

SQAI (SQL-Based AI Analysis) is a platform that transforms natural language questions into intelligent SQL queries, executes them against your database, and generates insights with visualizations.

---

## Key Features

### **AI-Powered Query Generation**
- Uses OpenAI GPT-4 to understand natural language
- Generates optimised SQL queries specific to your schema

### **Semantic Schema Retrieval**
- Vector embeddings identify relevant tables automatically
- Cosine similarity search finds the right schema

### **Automatic Analysis & Visualization**
- Mistral 7B generates insights from query results
- Auto-generates Python/matplotlib visualization code

### **Schema Synchronization**
- Detects database schema changes
- Incremental updates (only changed tables are re-indexed)
- table descriptions generated via LLM

### **Multiple Interfaces**
- **CLI**: Command-line REPL
- **Streamlit Web UI**: Modern, interactive dashboard

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM (SQL)** | OpenAI GPT-4 |
| **LLM (Analysis)** | Mistral 7B (via Ollama) |
| **Vector Store** | Chroma DB |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Database** | MySQL 5.7+ |
| **Web UI** | Streamlit |
| **Python** | 3.9+ |

---

## Quick Start

### Prerequisites

```bash
# MySQL database running
# OpenAI API key with GPT-4 access
# Ollama running with Mistral model
ollama pull mistral
ollama serve
```

### Configuration (.env)

```bash
# Database
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database

# OpenAI (for SQL generation)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# Ollama (for analysis generation)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=mistral

# Vector Store
ENABLE_VECTOR_RETRIEVAL=true
VECTOR_DB_PATH=./schema_embeddings
VECTOR_TOP_K=3
SAMPLES_PER_TABLE=5

# Schema Sync
ENABLE_PERIODIC_SYNC=true
SCHEMA_SYNC_INTERVAL=3600

# Other
ENABLE_ANALYSIS=true
MAX_ANALYSIS_RECORDS=250
```

### Run CLI Interface

```bash
python applications/cli_app.py

```

### Run Streamlit Web UI

```bash
streamlit run applications/streamlit_app.py

```

---

## Component Documentation

### Core Modules

| Module | Purpose |
|--------|---------|
| `query_pipeline.py` | Main orchestration engine |
| `config.py` | Configuration management |
| `db_connector.py` | Database connection & execution |
| `vector_store.py` | Vector embeddings & search |

### Schema Management

| Module | Purpose |
|--------|---------|
| `schema_management/schema_inspector.py` | Extract schema metadata |
| `schema_management/schema_formatter.py` | Format schema for display |
| `schema_management/schema_document_generator.py` | Create embeddings |
| `schema_management/schema_retrieval.py` | Semantic search interface |
| `schema_management/schema_sync.py` | Schema change detection |

### LLM Generators

| Module | Purpose |
|--------|---------|
| `generators/query_generator.py` | SQL generation via LLM |
| `generators/analysis_generator.py` | Insights & visualization code via LLM |
| `generators/table_description_generator.py` | Table descriptions |

### User Interfaces

| Module | Purpose |
|--------|---------|
| `applications/cli_app.py` | CLI REPL interface |
| `applications/streamlit_app.py` | Web dashboard |
| `streamlit_app/ui.py` | UI components |
| `streamlit_app/models.py` | Pipeline manager (for web) |
| `streamlit_app/viz.py` | Visualization rendering |

---