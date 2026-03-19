from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st



def _kpi_value(df: pd.DataFrame, spec: Dict[str, Any]) -> str:
    kpi_type = spec.get("type")
    col = spec.get("column")

    if kpi_type == "row_count":
        return f"{len(df):,}"
    if kpi_type == "unique_count":
        if col == "__columns__":
            return f"{df.shape[1]:,}"
        if col in df.columns:
            return f"{df[col].nunique(dropna=True):,}"
    if kpi_type == "sum" and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        return f"{df[col].sum():,.2f}"
    if kpi_type == "avg" and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        return f"{df[col].mean():,.2f}"
    if kpi_type == "missing_count":
        if col in df.columns:
            return f"{df[col].isna().sum():,}"
        return f"{int(df.isna().sum().sum()):,}"
    return "N/A"



def _aggregate_df(df: pd.DataFrame, x: str, y: str | None, aggregation: str):
    if x not in df.columns:
        return None
    if aggregation == "count" or y is None or y not in df.columns:
        temp = df.groupby(x, dropna=False).size().reset_index(name="value")
        return temp
    if aggregation == "sum":
        temp = df.groupby(x, dropna=False)[y].sum().reset_index(name="value")
        return temp
    if aggregation == "mean":
        temp = df.groupby(x, dropna=False)[y].mean().reset_index(name="value")
        return temp
    return None



def chart_from_spec(df: pd.DataFrame, spec: Dict[str, Any]):
    chart_type = spec.get("chart_type")
    title = spec.get("title", "Chart")
    x = spec.get("x")
    y = spec.get("y")
    z = spec.get("z")
    color = spec.get("color")
    aggregation = spec.get("aggregation", "none")

    if chart_type == "histogram" and x in df.columns:
        return px.histogram(df, x=x, color=color if color in df.columns else None, title=title)

    if chart_type == "box" and y in df.columns:
        return px.box(df, y=y, color=color if color in df.columns else None, title=title)

    if chart_type == "scatter" and x in df.columns and y in df.columns:
        return px.scatter(df, x=x, y=y, color=color if color in df.columns else None, title=title)

    if chart_type == "3d_scatter" and x in df.columns and y in df.columns and z in df.columns:
        return px.scatter_3d(df, x=x, y=y, z=z, color=color if color in df.columns else None, title=title)

    if chart_type in {"bar", "line", "area", "pie"} and x in df.columns:
        if aggregation == "none" and y in df.columns:
            plot_df = df.copy()
            value_col = y
        else:
            plot_df = _aggregate_df(df, x, y, aggregation)
            value_col = "value"
        if plot_df is None or plot_df.empty:
            return None
        if chart_type == "bar":
            return px.bar(plot_df, x=x, y=value_col, color=color if color in plot_df.columns else None, title=title)
        if chart_type == "line":
            return px.line(plot_df, x=x, y=value_col, color=color if color in plot_df.columns else None, title=title)
        if chart_type == "area":
            return px.area(plot_df, x=x, y=value_col, color=color if color in plot_df.columns else None, title=title)
        if chart_type == "pie":
            return px.pie(plot_df, names=x, values=value_col, title=title)

    if chart_type == "heatmap":
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] >= 2:
            corr = numeric.corr(numeric_only=True)
            return px.imshow(corr, aspect="auto", title=title)
    return None



def render_bi_dashboard(
    df: pd.DataFrame,
    spec: Dict[str, Any],
    key_prefix: str = "bi_dashboard",
):
    st.markdown(
        """
        <style>
        .bi-card {background: linear-gradient(135deg,#0f172a,#1e293b); color: white; padding: 18px; border-radius: 16px;}
        .bi-sub {color: #cbd5e1;}
        .kpi-card {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px; box-shadow: 0 6px 18px rgba(15,23,42,0.06);}
        .kpi-label {font-size: 13px; color: #64748b; text-transform: uppercase; letter-spacing: .04em;}
        .kpi-value {font-size: 28px; font-weight: 700; color: #0f172a;}
        .insight-box {background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    title = spec.get("title", "AI BI Dashboard")
    subtitle = spec.get("subtitle", "")
    summary = spec.get("summary", "")

    st.markdown(
        f"""
        <div class="bi-card">
            <h1 style="margin-bottom: 4px;">{title}</h1>
            <div class="bi-sub">{subtitle}</div>
            <p style="margin-top: 12px; margin-bottom: 0;">{summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### KPI Overview")
    kpis: List[Dict[str, Any]] = spec.get("kpis", [])[:6]
    if kpis:
        cols = st.columns(len(kpis))
        for col, kpi in zip(cols, kpis):
            value = _kpi_value(df, kpi)
            col.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{kpi.get('label','KPI')}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Executive Insights")
    insights = spec.get("insights", [])
    if insights:
        for insight in insights:
            st.markdown(f"<div class='insight-box'>• {insight}</div>", unsafe_allow_html=True)
    else:
        st.info("No AI insights were generated.")

    charts = spec.get("charts", [])
    if charts:
        st.markdown("### Dashboard Visuals")
        for i in range(0, len(charts), 2):
            row = charts[i : i + 2]
            cols = st.columns(len(row))
            for j, (col, chart_spec) in enumerate(zip(cols, row)):
                with col:
                    fig = chart_from_spec(df, chart_spec)
                    if fig is not None:
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"{key_prefix}_chart_{i + j}",
                        )
                    else:
                        st.warning(f"Could not render chart: {chart_spec.get('title', 'Untitled')}")
