import streamlit as st
import pandas as pd
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.ui import (
    SessionState,
    QueryInput,
    ResultsDisplay,
    ErrorDisplay,
)
from streamlit_app.models import PipelineManager, VisualizationParser
from streamlit_app.viz import render_visualization

# Page config 
st.set_page_config(
    page_title="SQAI",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='12' fill='%2309090b'/><rect x='20' y='55' width='12' height='30' rx='2' fill='%23fff'/><rect x='38' y='35' width='12' height='50' rx='2' fill='%23fff'/><rect x='56' y='45' width='12' height='40' rx='2' fill='%23fff'/><rect x='74' y='25' width='12' height='60' rx='2' fill='%23fff'/></svg>",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600&family=Geist+Mono:wght@400;500&display=swap');

    /* ── DESIGN TOKENS (dark) ───────────────────────────────────────────────── */
    :root {
        --background:        #000000;
        --foreground:        #fafafa;
        --muted:             #111111;
        --muted-foreground:  #a1a1aa;
        --border:            #27272a;
        --ring:              #fafafa;
        --accent:            #18181b;
        --accent-foreground: #fafafa;
        --destructive:       #ef4444;
        --radius:            6px;
        --font-sans:         'Geist', ui-sans-serif, system-ui, sans-serif;
        --font-mono:         'Geist Mono', ui-monospace, monospace;
    }

    /* ── GLOBAL RESET ──────────────────────────────────────────────────────── */
    html, body {
        font-family: var(--font-sans) !important;
        color: var(--foreground) !important;
        background-color: var(--background) !important;
        -webkit-font-smoothing: antialiased;
    }
            
    [data-testid="stIconMaterial"] {
        display: none !important;
    }

    /* Force dark background everywhere */
    .stApp,
    .stApp > div,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > div,
    [data-testid="stVerticalBlock"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .main,
    .main > div {
        background-color: var(--background) !important;
        background: var(--background) !important;
    }

    /* Remove Streamlit chrome & sidebar toggle */
    #MainMenu, footer, header,
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] { display: none !important; }

    /* ── LAYOUT ─────────────────────────────────────────────────────────────── */
    .block-container {
        max-width: 100% !important;
        padding: 2.5rem 3rem 4rem !important;
        background: var(--background) !important;
    }

    @media (max-width: 768px) {
        .block-container { padding: 1.5rem 1.25rem 3rem !important; }
    }
            
            

    /* ── TYPOGRAPHY ─────────────────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
        color: var(--foreground) !important;
    }

    p, li, span, label, div {
        font-family: var(--font-sans) !important;
        color: var(--foreground) !important;
    }

    h1 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.03em !important;
    }

    h2 {
        font-size: 1.125rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        font-size: 0.9375rem !important;
        font-weight: 500 !important;
    }

    /* ── DIVIDERS ────────────────────────────────────────────────────────────── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── BUTTONS ─────────────────────────────────────────────────────────────── */
    .stButton > button,
    .stDownloadButton > button {
        font-family: var(--font-sans) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        background: var(--background) !important;
        color: #000000 !important;
        border: 1px solid var(--foreground) !important;
        border-radius: var(--radius) !important;
        padding: 0.5rem 1rem !important;
        height: auto !important;
        line-height: 1.5 !important;
        transition: opacity 0.15s ease !important;
        box-shadow: none !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        opacity: 0.75 !important;
        border-color: var(--foreground) !important;
    }

    /* ── TEXT INPUT / TEXTAREA ───────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        font-family: var(--font-sans) !important;
        font-size: 0.9375rem !important;
        background: var(--muted) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--foreground) !important;
        padding: 0.625rem 0.875rem !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border: 2px solid var(--foreground) !important;
        box-shadow: 0 0 0 3px rgba(250,250,250,0.08) !important;
        outline: none !important;
        background: var(--muted) !important;
    }

    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: var(--muted-foreground) !important;
    }

    .stTextInput > label,
    .stTextArea > label {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--foreground) !important;
        margin-bottom: 0.375rem !important;
    }

    /* ── SELECTBOX ───────────────────────────────────────────────────────────── */
    .stSelectbox > div > div {
        font-family: var(--font-sans) !important;
        font-size: 0.875rem !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        background: var(--muted) !important;
        color: var(--foreground) !important;
        box-shadow: none !important;
    }

    /* Selectbox dropdown */
    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background: var(--muted) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }

    [data-baseweb="option"] {
        background: var(--muted) !important;
        color: var(--foreground) !important;
    }

    [data-baseweb="option"]:hover {
        background: var(--accent) !important;
    }

    /* ── TABS ─────────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0 !important;
        padding: 0 !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-sans) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--muted-foreground) !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.625rem 1rem !important;
        margin-bottom: -1px !important;
        border-radius: 0 !important;
        transition: color 0.15s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--foreground) !important;
        background: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--foreground) !important;
        border-bottom-color: var(--foreground) !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding: 1.5rem 0 0 !important;
        background: transparent !important;
    }

    /* ── METRICS ──────────────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--muted) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem 1.25rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.03em !important;
        color: var(--foreground) !important;
        font-family: var(--font-sans) !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.8125rem !important;
        font-weight: 500 !important;
        color: var(--muted-foreground) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        font-family: var(--font-sans) !important;
    }

    [data-testid="stMetricDelta"] {
        color: var(--muted-foreground) !important;
    }

    /* ── EXPANDER ────────────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--foreground) !important;
        background: var(--muted) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.625rem 0.875rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;

        transition: background 0.15s ease !important;
    }
    .streamlit-expanderHeader:hover {
        background: var(--accent) !important;
    }
            
    .streamlit-expanderHeader svg {
        flex-shrink: 0 !important;
    }     

    .streamlit-expanderContent {
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius) var(--radius) !important;
        padding: 1rem !important;
        background: var(--muted) !important;
    }

    /* ── CODE BLOCKS ─────────────────────────────────────────────────────────── */
    .stCodeBlock, code, pre {
        font-family: var(--font-mono) !important;
        font-size: 0.8125rem !important;
        background: var(--muted) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--foreground) !important;
    }

    code { padding: 0.15em 0.4em !important; }

    /* ── ALERTS ──────────────────────────────────────────────────────────────── */
    .stAlert {
        font-family: var(--font-sans) !important;
        font-size: 0.875rem !important;
        border-radius: var(--radius) !important;
        border-width: 1px !important;
        padding: 0.75rem 1rem !important;
    }

    div[data-testid="stInfo"] {
        background: var(--muted) !important;
        border-color: var(--border) !important;
        color: var(--foreground) !important;
        border-radius: var(--radius) !important;
    }

    div[data-testid="stInfo"] p,
    div[data-testid="stInfo"] span {
        color: var(--foreground) !important;
    }

    div[data-testid="stWarning"] {
        background: #1c1500 !important;
        border-color: #3d2e00 !important;
        border-radius: var(--radius) !important;
    }

    div[data-testid="stError"] {
        background: #1c0000 !important;
        border-color: #3d0000 !important;
        border-radius: var(--radius) !important;
    }

    div[data-testid="stSuccess"] {
        background: #001c05 !important;
        border-color: #003d0e !important;
        border-radius: var(--radius) !important;
    }

    /* ── SPINNER ──────────────────────────────────────────────────────────────── */
    .stSpinner > div { border-top-color: var(--foreground) !important; }
    .stSpinner p {
        font-size: 0.875rem !important;
        color: var(--muted-foreground) !important;
        font-family: var(--font-sans) !important;
    }

    /* ── MARKDOWN TEXT ───────────────────────────────────────────────────────── */
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span {
        color: var(--foreground) !important;
    }

    .stMarkdown a {
        color: var(--foreground) !important;
        text-decoration: underline;
        text-underline-offset: 3px;
    }

    /* ── SCROLLBAR ───────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--muted-foreground); }

    /* ── COLUMN CONTAINERS ───────────────────────────────────────────────────── */
    [data-testid="stHorizontalBlock"],
    [data-testid="column"] {
        background: transparent !important;
    }

    /* ── DOWNLOAD BUTTON ─────────────────────────────────────────────────────── */
    .stDownloadButton > button {
        background: transparent !important;
        color: var(--foreground) !important;
        border: 1px solid var(--border) !important;
    }

    .stDownloadButton > button:hover {
        background: var(--muted) !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# global pipeline manager 
@st.cache_resource
def get_pipeline_manager() -> PipelineManager:
    manager = PipelineManager()

    if not manager.initialize():
        st.error(f"Pipeline initialization failed: {manager._init_error}")

    return manager



# The main applicaiton 

def main():

    SessionState.initialize()

    st.markdown("""
    <div style="
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #27272a;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    ">
        <span style="
            font-size: 1.125rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            color: #fafafa;
            font-family: 'Geist', ui-sans-serif, sans-serif;
        ">SQAI</span>
        <span style="
            font-size: 0.6875rem;
            font-weight: 500;
            color: #a1a1aa;
            background: #111111;
            border: 1px solid #27272a;
            border-radius: 9999px;
            padding: 0.125rem 0.5rem;
            font-family: 'Geist', ui-sans-serif, sans-serif;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        ">LLM-Powered SQL Analysis</span>
    </div>
    """, unsafe_allow_html=True)

    col_content = st.container()

    with col_content:
        query = QueryInput.render()

        if query:
            pipeline_manager = get_pipeline_manager()

            if not pipeline_manager._initialized:
                ErrorDisplay.render(
                    "Pipeline failed to initialize, check database credentials in .env"
                )
            else:
                with st.spinner("Processing your query..."):
                    result = pipeline_manager.execute_query(query)

                if result.get("success", False):
                    st.session_state.last_query = query
                    st.session_state.current_result = result
                    SessionState.add_to_history(query, result)
                    _render_results(result)

                elif "error" in result:
                    ErrorDisplay.render(result["error"])
                    SessionState.clear_state()

        elif st.session_state.current_result and st.session_state.last_query:
            st.markdown(f"""
            <div style="
                font-size: 0.8125rem;
                color: #a1a1aa;
                background: #111111;
                border: 1px solid #27272a;
                border-radius: 6px;
                padding: 0.625rem 0.875rem;
                margin-bottom: 1.25rem;
                font-family: 'Geist', ui-sans-serif, sans-serif;
            ">
                Last query &mdash;
                <span style="color: #fafafa; font-weight: 500;">
                    {st.session_state.last_query}
                </span>
            </div>
            """, unsafe_allow_html=True)
            _render_results(st.session_state.current_result)


def _section_label(text: str) -> None:
    st.markdown(f"""
    <p style="
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #a1a1aa;
        margin: 0 0 0.875rem;
        font-family: 'Geist', ui-sans-serif, sans-serif;
    ">{text}</p>
    """, unsafe_allow_html=True)


def _render_results(result: Dict[str, Any]) -> None:
    # Render query results with insights and visualization

    if "error" in result:
        ErrorDisplay.render(result["error"])
        return

    insights_json      = result.get("analysis", "")
    visualization_json = result.get("visualization_code", "")
    execution_results  = result.get("execution_results", {})

    result_df = execution_results.get("analysis_df")
    if result_df is None or result_df.empty:
        st.warning("No results returned from query.")
        return

    tab_insights, tab_data, tab_details = st.tabs(["Insights", "Data", "Details"])

    # Insights tab
    with tab_insights:
        if insights_json:
            ResultsDisplay.render_insights(insights_json)
            st.divider()

        viz_spec = VisualizationParser.parse(visualization_json)

        if viz_spec is not None:
            _section_label("Visualization")
            is_valid, error_msg = VisualizationParser.validate(viz_spec)

            if is_valid:
                try:
                    render_visualization(result_df, viz_spec)
                except Exception as e:
                    st.error(f"Visualization error: {str(e)}")
            else:
                st.error(f"Invalid visualization spec: {error_msg}")
        else:
            ResultsDisplay.render_visualization_placeholder(has_viz=False)

    # Data tab
    with tab_data:
        _section_label("Query Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{len(result_df):,}")
        with col2:
            st.metric("Columns", len(result_df.columns))
            
        st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

        if len(result_df) > 50:
            st.info(f"Showing first 50 of {len(result_df):,} rows.")
            st.dataframe(result_df.head(50), width="stretch")
        else:
            st.dataframe(result_df, width="stretch")

        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="query_results.csv",
            mime="text/csv",
        )

    # Details tab
    with tab_details:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            _section_label("SQL Queries")
            sql_queries = result.get("sql_queries", {})

            if sql_queries.get("analysis_query"):
                with st.expander("Analysis Query"):
                    st.code(sql_queries["analysis_query"], language="sql")

            if sql_queries.get("visualization_query"):
                with st.expander("Visualization Query"):
                    st.code(sql_queries["visualization_query"], language="sql")

        with col2:
            _section_label("Performance")
            timing = result.get("timing", {})

            if timing:
                stage_names = {
                    "schema_retrieval":    "Schema Retrieval",
                    "context_formatting":  "Context Formatting",
                    "sql_generation":      "SQL Generation",
                    "sql_execution":       "Query Execution",
                    "analysis_generation": "Analysis Generation",
                }

                metrics_data: Dict[str, list] = {"Stage": [], "Time (s)": []}
                for key, label in stage_names.items():
                    if key in timing:
                        metrics_data["Stage"].append(label)
                        metrics_data["Time (s)"].append(round(timing[key], 2))

                if metrics_data["Stage"]:
                    st.dataframe(
                        pd.DataFrame(metrics_data),
                        width='stretch',
                        hide_index=True,
                    )
                    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
                    st.metric("Total Time", f"{sum(timing.values()):.2f}s")

        st.divider()
        ResultsDisplay.render_metadata(result)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page.")