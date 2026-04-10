import streamlit as st
from typing import Optional, Dict, Any


class SessionState:

    @staticmethod
    def initialize() -> None:

        if "query_history" not in st.session_state:
            st.session_state.query_history = []
        
        if "current_result" not in st.session_state:
            st.session_state.current_result = None
        
        if "error_message" not in st.session_state:
            st.session_state.error_message = None
        
        if "last_query" not in st.session_state:
            st.session_state.last_query = ""
        
        if "loading" not in st.session_state:
            st.session_state.loading = False

    @staticmethod
    def add_to_history(query: str, result: Dict[str, Any]) -> None:
        #Add query and result to history
        st.session_state.query_history.append({
            "query": query,
            "result": result,
        })

        if len(st.session_state.query_history) > 10:
            st.session_state.query_history = st.session_state.query_history[-10:]

    @staticmethod
    def clear_state() -> None:
        st.session_state.error_message = None

class QueryInput:

    @staticmethod
    def render() -> Optional[str]:
        
        query = st.text_area(
            "What would you like to analyse?",
            height=80,
            placeholder="Your Query...",
            key="query_input",
        )
        
        # Button row
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            submit = st.button("Analyse", use_container_width=True, type="primary")
        
        with col2:
            clear = st.button("Clear", use_container_width=True)
        
        with col3:
            st.markdown("")
        
        if clear:
            st.session_state.query_input = ""
            st.rerun()
        
        if submit and query.strip():
            return query.strip()
        
        return None


class ResultsDisplay:

    @staticmethod
    def render_insights(insights: str) -> None:
        st.subheader("Key Insights", divider="green")
        
        try:
            import json
            insights_data = json.loads(insights)
            
            if isinstance(insights_data, list):
                for i, item in enumerate(insights_data, 1):
                    if isinstance(item, dict) and "insight" in item:
                        st.markdown(f"**{i}.** {item['insight']}")
                    else:
                        st.markdown(f"**{i}.** {item}")
            else:
                st.markdown(insights)
        except (json.JSONDecodeError, TypeError):
            st.markdown(insights)

    @staticmethod
    def render_visualization_placeholder(has_viz: bool) -> None:
        if not has_viz:
            st.info("No visualization needed for this query.")
        else:
            st.info("Visualization loading...")

    @staticmethod
    def render_metadata(result: Dict[str, Any]) -> None:
        with st.expander("Query Details"):
            col1, col2, col3, col4 = st.columns(4)
            
            timing = result.get("timing", {})
            
            if "retrieved_tables" in result:
                with col1:
                    st.metric(
                        "Tables Found",
                        len(result.get("retrieved_tables", []))
                    )
            
            if "sql_generation" in timing:
                with col2:
                    st.metric(
                        "SQL Generation",
                        f"{timing['sql_generation']:.1f}s"
                    )
            
            if "sql_execution" in timing:
                with col3:
                    st.metric(
                        "Query Execution",
                        f"{timing['sql_execution']:.1f}s"
                    )
            
            if "analysis_generation" in timing:
                with col4:
                    st.metric(
                        "Analysis Generation",
                        f"{timing['analysis_generation']:.1f}s"
                    )


class ErrorDisplay:
    @staticmethod
    def render(error_msg: str) -> None:
        st.error(f"{error_msg}")