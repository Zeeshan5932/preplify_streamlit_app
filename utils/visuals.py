from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return None
    corr = numeric.corr(numeric_only=True)
    return px.imshow(corr, text_auto=False, aspect="auto", title="Correlation Heatmap")



def histogram_chart(df: pd.DataFrame, column: str):
    return px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")



def box_chart(df: pd.DataFrame, column: str, color: Optional[str] = None):
    return px.box(df, y=column, color=color, title=f"Box Plot of {column}")



def scatter_2d(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, size: Optional[str] = None):
    return px.scatter(df, x=x, y=y, color=color, size=size, title=f"{x} vs {y}")



def scatter_3d(df: pd.DataFrame, x: str, y: str, z: str, color: Optional[str] = None, size: Optional[str] = None):
    return px.scatter_3d(df, x=x, y=y, z=z, color=color, size=size, title=f"3D Scatter: {x}, {y}, {z}")



def pca_3d(df: pd.DataFrame, color: Optional[str] = None):
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 3 or numeric.shape[0] < 3:
        return None
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=3, random_state=42)),
    ]
    pipe = Pipeline(steps)
    comps = pipe.fit_transform(numeric)
    pca_df = pd.DataFrame(comps, columns=["PC1", "PC2", "PC3"], index=df.index)
    if color and color in df.columns:
        pca_df[color] = df[color].astype(str)
    return px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color if color in pca_df.columns else None,
        title="3D PCA Projection",
    )



def cluster_3d(df: pd.DataFrame, n_clusters: int = 4):
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 3 or numeric.shape[0] < max(4, n_clusters):
        return None
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=3, random_state=42)),
    ]
    pipe = Pipeline(steps)
    comps = pipe.fit_transform(numeric)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(comps)
    cluster_df = pd.DataFrame(comps, columns=["PC1", "PC2", "PC3"])
    cluster_df["Cluster"] = labels.astype(str)
    return px.scatter_3d(
        cluster_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        title="3D Cluster Explorer",
    )



def missing_bar(df: pd.DataFrame):
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return None
    plot_df = missing.reset_index()
    plot_df.columns = ["column", "missing_count"]
    return px.bar(plot_df, x="column", y="missing_count", title="Missing Values by Column")



def numeric_pair_candidates(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()
