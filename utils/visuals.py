from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st



def render_visual_section(df: pd.DataFrame) -> None:
    st.subheader("Visualize your data")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    chart_type = st.selectbox(
        "Chart type",
        ["Histogram", "Box plot", "Scatter plot", "Bar chart", "Correlation heatmap"],
    )

    if chart_type == "Histogram":
        if not numeric_cols:
            st.warning("No numeric columns found for histogram.")
            return
        col = st.selectbox("Numeric column", numeric_cols)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box plot":
        if not numeric_cols:
            st.warning("No numeric columns found for box plot.")
            return
        y_col = st.selectbox("Numeric column", numeric_cols)
        color_col = st.selectbox("Optional category", [None] + categorical_cols)
        fig = px.box(df, y=y_col, color=color_col)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter plot":
        if len(numeric_cols) < 2:
            st.warning("At least two numeric columns are needed for a scatter plot.")
            return
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X-axis", numeric_cols)
        y_col = c2.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        color_col = c3.selectbox("Color", [None] + categorical_cols)
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar chart":
        if not categorical_cols:
            st.warning("No categorical columns found for bar chart.")
            return
        col = st.selectbox("Category column", categorical_cols)
        counts = df[col].astype(str).value_counts(dropna=False).reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Correlation heatmap":
        if len(numeric_cols) < 2:
            st.warning("At least two numeric columns are needed for a correlation heatmap.")
            return
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
