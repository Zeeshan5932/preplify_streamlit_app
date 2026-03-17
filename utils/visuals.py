from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def render_visual_lab(df: pd.DataFrame) -> None:
    tab1, tab2, tab3, tab4 = st.tabs([
        "2D charts",
        "3D explorer",
        "Missingness",
        "Anomaly finder",
    ])

    with tab1:
        render_2d_charts(df)
    with tab2:
        render_3d_explorer(df)
    with tab3:
        render_missingness(df)
    with tab4:
        render_anomaly_finder(df)


def render_2d_charts(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    chart_type = st.selectbox(
        "Choose a chart",
        ["Histogram", "Box plot", "Scatter plot", "Bar chart", "Correlation heatmap"],
        key="2d_chart_type",
    )

    if chart_type == "Histogram":
        if not numeric_cols:
            st.warning("No numeric columns found for histogram.")
            return
        col = st.selectbox("Numeric column", numeric_cols, key="hist_col")
        bins = st.slider("Bins", 10, 100, 30)
        color_col = st.selectbox("Optional color", [None] + categorical_cols, key="hist_color")
        fig = px.histogram(df, x=col, color=color_col, nbins=bins)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box plot":
        if not numeric_cols:
            st.warning("No numeric columns found for box plot.")
            return
        y_col = st.selectbox("Numeric column", numeric_cols, key="box_y")
        x_col = st.selectbox("Optional category", [None] + categorical_cols, key="box_x")
        fig = px.box(df, x=x_col, y=y_col, points="outliers")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter plot":
        if len(numeric_cols) < 2:
            st.warning("At least two numeric columns are needed for a scatter plot.")
            return
        c1, c2, c3, c4 = st.columns(4)
        x_col = c1.selectbox("X-axis", numeric_cols, key="scatter_x")
        y_col = c2.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_y")
        color_col = c3.selectbox("Color", [None] + categorical_cols, key="scatter_color")
        size_col = c4.selectbox("Size", [None] + numeric_cols, key="scatter_size")
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, hover_data=df.columns[:8])
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar chart":
        if not categorical_cols:
            st.warning("No categorical columns found for bar chart.")
            return
        col = st.selectbox("Category column", categorical_cols, key="bar_col")
        top_n = st.slider("Top categories", 5, 30, 10)
        counts = df[col].astype(str).value_counts(dropna=False).head(top_n).reset_index()
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


def render_3d_explorer(df: pd.DataFrame) -> None:
    st.markdown("### 3D visualization studio")
    mode = st.radio(
        "3D mode",
        ["3D scatter", "3D PCA projection", "3D clusters"],
        horizontal=True,
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    if mode == "3D scatter":
        if len(numeric_cols) < 3:
            st.warning("You need at least three numeric columns for a true 3D scatter plot.")
            return
        c1, c2, c3, c4, c5 = st.columns(5)
        x_col = c1.selectbox("X", numeric_cols, key="3d_x")
        y_col = c2.selectbox("Y", numeric_cols, index=1, key="3d_y")
        z_col = c3.selectbox("Z", numeric_cols, index=2, key="3d_z")
        color_col = c4.selectbox("Color", [None] + categorical_cols + numeric_cols, key="3d_color")
        size_col = c5.selectbox("Size", [None] + numeric_cols, key="3d_size")
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col, size=size_col, opacity=0.75)
        st.plotly_chart(fig, use_container_width=True)

    elif mode == "3D PCA projection":
        transformed, feature_names = _numeric_ready_matrix(df)
        if transformed is None or transformed.shape[1] < 3:
            st.warning("Need enough usable numeric information to compute 3 PCA components.")
            return
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(transformed)
        pca = PCA(n_components=3, random_state=42)
        comps = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(comps, columns=["PC1", "PC2", "PC3"])
        pca_df["row_id"] = np.arange(len(pca_df))
        color_col = st.selectbox("Color by", [None] + categorical_cols + numeric_cols, key="pca_color")
        if color_col is not None:
            pca_df[color_col] = df[color_col].values
        fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color=color_col, hover_data=["row_id"])
        st.plotly_chart(fig, use_container_width=True)

        explained = pca.explained_variance_ratio_ * 100
        st.caption(
            f"Explained variance: PC1 {explained[0]:.1f}% | PC2 {explained[1]:.1f}% | PC3 {explained[2]:.1f}%"
        )
        st.write("Top transformed input features used:", feature_names[:10])

    elif mode == "3D clusters":
        transformed, _ = _numeric_ready_matrix(df)
        if transformed is None or transformed.shape[1] < 3:
            st.warning("Need enough usable numeric information to create 3D clusters.")
            return
        c1, c2 = st.columns(2)
        n_clusters = c1.slider("Number of clusters", 2, 8, 3)
        random_state = c2.number_input("Random state", min_value=0, value=42, step=1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(transformed)
        pca = PCA(n_components=3, random_state=int(random_state))
        comps = pca.fit_transform(X_scaled)
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(random_state)).fit_predict(X_scaled)

        cluster_df = pd.DataFrame(comps, columns=["PC1", "PC2", "PC3"])
        cluster_df["cluster"] = labels.astype(str)
        fig = px.scatter_3d(cluster_df, x="PC1", y="PC2", z="PC3", color="cluster", opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)

        cluster_sizes = pd.Series(labels).value_counts().sort_index().rename_axis("cluster").reset_index(name="count")
        st.dataframe(cluster_sizes, use_container_width=True)


def render_missingness(df: pd.DataFrame) -> None:
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        st.success("No missing values found in this dataset.")
        return

    fig = px.bar(
        missing_pct.reset_index(),
        x="index",
        y=0,
        labels={"index": "column", 0: "missing %"},
        title="Missing values by column",
    )
    st.plotly_chart(fig, use_container_width=True)

    sample_rows = min(150, len(df))
    missing_matrix = df.head(sample_rows).isna().astype(int)
    fig2 = px.imshow(missing_matrix.T, aspect="auto", labels={"x": "row", "y": "column", "color": "missing"})
    st.plotly_chart(fig2, use_container_width=True)


def render_anomaly_finder(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2 or len(numeric_df) < 10:
        st.warning("Anomaly detection needs at least 2 numeric columns and around 10 rows.")
        return

    c1, c2 = st.columns(2)
    contamination = c1.slider("Expected anomaly share", 0.01, 0.30, 0.05, step=0.01)
    random_state = c2.number_input("Random state", min_value=0, value=42, step=1, key="anomaly_rs")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(numeric_df)
    model = IsolationForest(contamination=contamination, random_state=int(random_state))
    flags = model.fit_predict(X)
    scores = model.decision_function(X)

    result = df.copy()
    result["anomaly_flag"] = np.where(flags == -1, "anomaly", "normal")
    result["anomaly_score"] = scores

    st.metric("Detected anomalies", int((flags == -1).sum()))

    if X.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=int(random_state))
        coords = pca.fit_transform(StandardScaler().fit_transform(X))
        plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "anomaly_flag": result["anomaly_flag"]})
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="anomaly_flag", opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(result.sort_values("anomaly_score").head(30), use_container_width=True)
    st.download_button(
        "Download anomaly_results.csv",
        data=result.to_csv(index=False),
        file_name="anomaly_results.csv",
        mime="text/csv",
    )


def _numeric_ready_matrix(df: pd.DataFrame):
    if df.empty:
        return None, []
    X = pd.get_dummies(df.copy(), dummy_na=True)
    if X.empty:
        return None, []
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.empty:
        return None, []
    imputer = SimpleImputer(strategy="median")
    transformed = imputer.fit_transform(X)
    return transformed, X.columns.tolist()
