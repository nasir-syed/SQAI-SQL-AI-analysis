import json
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import io

class VisualizationRenderer:

    def __init__(self, data: pd.DataFrame, spec: Dict[str, Any]):
        self.data = data
        self.spec = spec
        self.chart_type = spec.get("chart_type", "bar")
        self.title = spec.get("title", "Chart")
        self._validation_errors = []

    def _validate(self) -> bool:
        self._validation_errors = []

        if self.data is None or self.data.empty:
            self._validation_errors.append("Data is empty")
            return False

        x_axis = self.spec.get("x_axis", {})
        y_axis = self.spec.get("y_axis", {})
        x_col = x_axis.get("column")
        y_col = y_axis.get("column")

        if self.chart_type == "histogram":
            if not y_col:
                self._validation_errors.append("Histogram requires y_axis.column")
                return False
            if y_col not in self.data.columns:
                self._validation_errors.append(f"Column '{y_col}' not found in data")
                return False
        else:
            if not x_col or not y_col:
                self._validation_errors.append(
                    f"{self.chart_type} requires both x_axis.column and y_axis.column"
                )
                return False

            if x_col not in self.data.columns:
                self._validation_errors.append(f"X-axis column '{x_col}' not found in data")
                return False

            if y_col not in self.data.columns:
                self._validation_errors.append(f"Y-axis column '{y_col}' not found in data")
                return False

        if self.chart_type in ["bar", "barh", "box", "pie"]:
            unique_x = self.data[x_col].nunique() if x_col in self.data.columns else 0
            if unique_x < 3:
                self._validation_errors.append(
                    f"{self.chart_type} requires at least 3 unique values on x-axis, got {unique_x}"
                )
                return False
        elif self.chart_type in ["line", "area"]:
            if len(self.data) < 3:
                self._validation_errors.append(
                    f"{self.chart_type} requires at least 3 data points, got {len(self.data)}"
                )
                return False

        return True

    def _to_numeric(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _to_categorical(self, series: pd.Series) -> pd.Series:
        return series.astype(str)

    def _prepare_categorical_data(
        self, plot_data: pd.DataFrame, cat_col: str, num_col: str
    ) -> pd.DataFrame:
        top_n = self.spec.get("top_n")
        if top_n and isinstance(top_n, int):
            plot_data = plot_data.nlargest(top_n, num_col)

        trunc = self.spec.get("label_truncate_length")
        if trunc and isinstance(trunc, int):
            plot_data = plot_data.copy()
            plot_data[cat_col] = plot_data[cat_col].apply(
                lambda x: (str(x)[:trunc] + "…") if len(str(x)) > trunc else str(x)
            )

        return plot_data

    def render(self) -> Optional[plt.Figure]:
        try:
            if not self._validate():
                error_msg = "; ".join(self._validation_errors)
                st.error(f"Visualization validation failed: {error_msg}")
                return None

            dispatch = {
                "bar": self._render_bar,
                "barh": self._render_barh,
                "line": self._render_line,
                "scatter": self._render_scatter,
                "histogram": self._render_histogram,
                "pie": self._render_pie,
                "box": self._render_box,
                "heatmap": self._render_heatmap,
                "area": self._render_area,
            }

            renderer_fn = dispatch.get(self.chart_type)
            if renderer_fn is None:
                st.error(f"Unknown chart type: {self.chart_type}")
                return None

            return renderer_fn()

        except Exception as e:
            st.error(f"Visualization rendering failed: {str(e)}")
            return None

    def _get_axis_config(self) -> tuple:
        x_axis = self.spec.get("x_axis", {})
        y_axis = self.spec.get("y_axis", {})

        x_col = x_axis.get("column")
        y_col = y_axis.get("column")
        x_label = x_axis.get("label", x_col or "X")
        y_label = y_axis.get("label", y_col or "Y")

        return x_col, y_col, x_label, y_label

    def _get_styling(self) -> Dict[str, Any]:
        styling = self.spec.get("styling", {})
        figure_width = styling.get("figure_width", 10)
        figure_height = styling.get("figure_height", 5)
        
        figure_width = min(figure_width, 10)
        figure_height = min(figure_height, 6)
        
        return {
            "color_palette": styling.get("color_palette", "viridis"),
            "marker_size": styling.get("marker_size", 6),
            "line_width": styling.get("line_width", 1.5),
            "alpha": styling.get("alpha", 0.8),
            "grid": styling.get("grid", True),
            "rotation_x_labels": styling.get("rotation_x_labels", 0),
            "figure_width": figure_width,
            "figure_height": figure_height,
        }

    def _apply_styling(self, ax: plt.Axes, styling: Dict[str, Any]) -> None:
        if styling.get("grid"):
            ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    def _render_bar(self) -> plt.Figure:
        x_col, y_col, x_label, y_label = self._get_axis_config()
        styling = self._get_styling()

        fig, ax = plt.subplots(figsize=(styling["figure_width"], styling["figure_height"]))

        try:
            plot_data = self.data[[x_col, y_col]].copy()
            plot_data[y_col] = self._to_numeric(plot_data[y_col])
            plot_data[x_col] = self._to_categorical(plot_data[x_col])
            plot_data = plot_data.dropna()

            if len(plot_data) < 3:
                raise ValueError(f"Not enough data points after cleanup (got {len(plot_data)}, need 3+)")

            plot_data = plot_data.sort_values(by=y_col, ascending=False)
            plot_data = self._prepare_categorical_data(plot_data, x_col, y_col)

            colors = plt.cm.get_cmap(styling["color_palette"])(
                np.linspace(0, 1, len(plot_data))
            )
            ax.bar(plot_data[x_col].astype(str), plot_data[y_col], color=colors, alpha=styling["alpha"])

            ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            plt.xticks(rotation=styling["rotation_x_labels"], ha="right")

            self._apply_styling(ax, styling)
            plt.tight_layout()
            return fig
        except Exception as e:
            plt.close(fig)
            st.error(f"Bar chart rendering error: {str(e)}")
            return None

    def _render_barh(self) -> plt.Figure:
        x_col, y_col, x_label, y_label = self._get_axis_config()
        styling = self._get_styling()

        fig, ax = plt.subplots(figsize=(styling["figure_width"], styling["figure_height"]))

        try:
            plot_data = self.data[[x_col, y_col]].copy()
            plot_data[y_col] = self._to_numeric(plot_data[y_col])
            plot_data[x_col] = self._to_categorical(plot_data[x_col])
            plot_data = plot_data.dropna()

            if len(plot_data) < 3:
                raise ValueError(f"Not enough data points after cleanup (got {len(plot_data)}, need 3+)")

            plot_data = self._prepare_categorical_data(plot_data, x_col, y_col)
            plot_data = plot_data.sort_values(by=y_col, ascending=True) 

            colors = plt.cm.get_cmap(styling["color_palette"])(
                np.linspace(0, 1, len(plot_data))
            )
            
            ax.barh(plot_data[x_col].astype(str), plot_data[y_col], color=colors, alpha=styling["alpha"])

            ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)
            ax.set_xlabel(y_label, fontsize=11)  
            ax.set_ylabel(x_label, fontsize=11)  

            self._apply_styling(ax, styling)
            plt.tight_layout()
            return fig
        except Exception as e:
            plt.close(fig)
            st.error(f"Horizontal bar chart rendering error: {str(e)}")
            return None

    def _render_line(self) -> plt.Figure:
        x_col, y_col, x_label, y_label = self._get_axis_config()
        styling = self._get_styling()

        fig, ax = plt.subplots(figsize=(styling["figure_width"], styling["figure_height"]))

        try:
            plot_data = self.data[[x_col, y_col]].copy()
            plot_data[y_col] = self._to_numeric(plot_data[y_col])
            plot_data = plot_data.dropna()

            if len(plot_data) < 3:
                raise ValueError(f"Not enough data points (got {len(plot_data)}, need 3+)")

            plot_data = plot_data.sort_values(by=x_col)

            ax.plot(
                plot_data[x_col].astype(str),
                plot_data[y_col],
                linewidth=styling["line_width"],
                marker="o",
                markersize=styling["marker_size"],
                alpha=styling["alpha"],
                color=plt.cm.get_cmap(styling["color_palette"])(0.5),
            )

            ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            plt.xticks(rotation=styling["rotation_x_labels"], ha="right")

            self._apply_styling(ax, styling)
            plt.tight_layout()
            return fig
        except Exception as e:
            plt.close(fig)
            st.error(f"Line chart rendering error: {str(e)}")
            return None

    def _render_scatter(self) -> plt.Figure:
        x_col, y_col, x_label, y_label = self._get_axis_config()
        styling = self._get_styling()

        fig, ax = plt.subplots(figsize=(styling["figure_width"], styling["figure_height"]))

        try:
            plot_data = self.data[[x_col, y_col]].copy()
            plot_data[x_col] = self._to_numeric(plot_data[x_col])
            plot_data[y_col] = self._to_numeric(plot_data[y_col])
            plot_data = plot_data.dropna()

            if len(plot_data) < 3:
                raise ValueError(f"Not enough data points (got {len(plot_data)}, need 3+)")

            ax.scatter(
                plot_data[x_col],
                plot_data[y_col],
                s=styling["marker_size"] * 30,
                alpha=styling["alpha"],
                c=range(len(plot_data)),
                cmap=styling["color_palette"],
            )

            ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)

            self._apply_styling(ax, styling)
            plt.tight_layout()
            return fig
        except Exception as e:
            plt.close(fig)
            st.error(f"Scatter plot rendering error: {str(e)}")
            return None

    def _render_histogram(self) -> plt.Figure:
        _, y_col, _, y_label = self._get_axis_config()
        styling = self._get_styling()

        fig, ax = plt.subplots(figsize=(styling["figure_width"], styling["figure_height"]))

        try:
            values = self._to_numeric(self.data[y_col]).dropna()

            if len(values) < 10:
                raise ValueError(
                    f"Not enough data points for meaningful histogram (got {len(values)}, need 10+)"
                )

            ax.hist(
                values,
                bins=20,
                color=plt.cm.get_cmap(styling["color_palette"])(0.5),
                alpha=styling["alpha"],
                edgecolor="black",
            )

            ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)
            ax.set_xlabel(y_label, fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)

            self._apply_styling(ax, styling)
            plt.tight_layout()
            return fig
        except Exception as e:
            plt.close(fig)
            st.error(f"Histogram rendering error: {str(e)}")
            return None

    def _render_pie(self) -> plt.Figure:
        x_col, y_col, _, _ = self._get_axis_config()
        styling = self._get_styling()

        fig, ax = plt.subplots(figsize=(10, 8))

        try:
            plot_data = self.data[[x_col, y_col]].copy()
            plot_data[x_col] = self._to_categorical(plot_data[x_col])
            plot_data[y_col] = self._to_numeric(plot_data[y_col])
            plot_data = plot_data.dropna()

            num_categories = plot_data[x_col].nunique()
            if num_categories < 3 or num_categories > 7:
                raise ValueError(f"Pie chart needs 3–7 categories, got {num_categories}")

            pie_data = plot_data.groupby(x_col)[y_col].sum()
            colors = plt.cm.get_cmap(styling["color_palette"])(
                np.linspace(0, 1, len(pie_data))
            )

            wedges, texts, autotexts = ax.pie(
                pie_data,
                labels=pie_data.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )

            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")

            ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)
            plt.tight_layout()
            return fig
        except Exception as e:
            plt.close(fig)
            st.error(f"Pie chart rendering error: {str(e)}")
            return None

    def _render_box(self) -> plt.Figure:
        x_col, y_col, x_label, y_label = self._get_axis_config()
        styling = self._get_styling()

        fig, ax = plt.subplots(figsize=(styling["figure_width"], styling["figure_height"]))

        try:
            plot_data = self.data[[x_col, y_col]].copy()
            plot_data[x_col] = self._to_categorical(plot_data[x_col])
            plot_data[y_col] = self._to_numeric(plot_data[y_col])
            plot_data = plot_data.dropna()

            num_groups = plot_data[x_col].nunique()
            if num_groups < 3:
                raise ValueError(f"Box plot needs at least 3 groups, got {num_groups}")

            grouped = [group[y_col].dropna() for _, group in plot_data.groupby(x_col)]
            ax.boxplot(grouped, labels=plot_data[x_col].unique())
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)
            plt.xticks(rotation=styling["rotation_x_labels"], ha="right")

            self._apply_styling(ax, styling)
            plt.tight_layout()
            return fig
        except Exception as e:
            plt.close(fig)
            st.error(f"Box plot rendering error: {str(e)}")
            return None

    def _render_heatmap(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 8))

        try:
            numeric_data = self.data.select_dtypes(include=[np.number]).copy()

            if numeric_data.empty:
                raise ValueError("Heatmap requires at least one numeric column")

            numeric_data = numeric_data.fillna(0)

            im = ax.imshow(numeric_data, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(numeric_data.columns)))
            ax.set_yticks(range(len(numeric_data.index)))
            ax.set_xticklabels(numeric_data.columns, rotation=45, ha="right")
            ax.set_yticklabels(numeric_data.index)
            ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Value", fontsize=11)

            plt.tight_layout()
            return fig
        except Exception as e:
            plt.close(fig)
            st.error(f"Heatmap rendering error: {str(e)}")
            return None

    def _render_area(self) -> plt.Figure:
        x_col, y_col, x_label, y_label = self._get_axis_config()
        styling = self._get_styling()

        fig, ax = plt.subplots(figsize=(styling["figure_width"], styling["figure_height"]))

        try:
            plot_data = self.data[[x_col, y_col]].copy()
            plot_data[y_col] = self._to_numeric(plot_data[y_col])
            plot_data = plot_data.dropna()

            if len(plot_data) < 3:
                raise ValueError(f"Not enough data points (got {len(plot_data)}, need 3+)")

            plot_data = plot_data.sort_values(by=x_col)
            x_values = plot_data[x_col].astype(str)
            y_values = plot_data[y_col]

            ax.fill_between(
                range(len(plot_data)),
                y_values,
                alpha=styling["alpha"],
                color=plt.cm.get_cmap(styling["color_palette"])(0.5),
            )
            ax.plot(
                range(len(plot_data)),
                y_values,
                linewidth=styling["line_width"],
                color=plt.cm.get_cmap(styling["color_palette"])(0.7),
            )
            ax.set_xticks(range(len(plot_data)))
            ax.set_xticklabels(x_values, rotation=styling["rotation_x_labels"], ha="right")

            ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)

            self._apply_styling(ax, styling)
            plt.tight_layout()
            return fig
        except Exception as e:
            plt.close(fig)
            st.error(f"Area chart rendering error: {str(e)}")
            return None


def render_visualization(data: pd.DataFrame, viz_spec: Dict[str, Any]) -> None:
    try:
        if viz_spec is None or not isinstance(viz_spec, dict):
            st.info("This query is best answered with text rather than a visualization")
            return

        renderer = VisualizationRenderer(data, viz_spec)
        fig = renderer.render()

        if fig:
            st.pyplot(fig, width='stretch')
            plt.close(fig)

            if st.button("Download Chart as PNG", key="download_chart"):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    label="Download Chart",
                    data=buf,
                    file_name="chart.png",
                    mime="image/png",
                    key="download_chart_btn"
                )
        else:
            st.warning("Could not render chart with provided specification")

    except Exception as e:
        st.error(f"Failed to render visualization: {str(e)}")