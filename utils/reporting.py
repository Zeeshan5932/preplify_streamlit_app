from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from preplify import data_report, recommend_preprocessing


def build_dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2))

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": missing_total,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "memory_mb": round(memory_mb, 2),
        "health_score": compute_health_score(df),
    }


def build_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_%": round(float(series.isna().mean() * 100), 2),
                "unique_values": int(series.nunique(dropna=False)),
                "sample": _safe_sample(series),
            }
        )
    return pd.DataFrame(rows)


def compute_health_score(df: pd.DataFrame) -> int:
    rows, cols = df.shape
    if rows == 0 or cols == 0:
        return 0

    missing_ratio = float(df.isna().sum().sum()) / max(rows * cols, 1)
    duplicate_ratio = float(df.duplicated().sum()) / max(rows, 1)
    constant_cols = int(df.nunique(dropna=False).le(1).sum())
    constant_ratio = constant_cols / max(cols, 1)

    penalty = (missing_ratio * 45) + (duplicate_ratio * 30) + (constant_ratio * 25)
    score = max(0, min(100, round(100 - penalty * 100)))
    return int(score)


def generate_smart_insights(df: pd.DataFrame) -> List[str]:
    insights: List[str] = []
    profile = build_dataset_profile(df)

    if profile["missing_values"] > 0:
        top_missing = df.isna().mean().sort_values(ascending=False)
        top_missing = top_missing[top_missing > 0].head(3)
        parts = [f"{col} ({val * 100:.1f}%)" for col, val in top_missing.items()]
        insights.append("Highest missingness appears in: " + ", ".join(parts) + ".")

    if profile["duplicate_rows"] > 0:
        insights.append(f"Dataset contains {profile['duplicate_rows']} duplicate rows that may distort training.")

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        skewness = numeric_df.skew(numeric_only=True).abs().sort_values(ascending=False)
        skewness = skewness[skewness > 1.0]
        if not skewness.empty:
            top_skew = ", ".join([f"{col} ({val:.2f})" for col, val in skewness.head(3).items()])
            insights.append(f"Highly skewed numeric columns detected: {top_skew}.")

        corr = numeric_df.corr(numeric_only=True).abs()
        if len(corr.columns) > 1:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_pairs = upper.stack().sort_values(ascending=False)
            high_pairs = high_pairs[high_pairs > 0.85]
            if not high_pairs.empty:
                pair_names = []
                for (a, b), value in high_pairs.head(3).items():
                    pair_names.append(f"{a} ↔ {b} ({value:.2f})")
                insights.append("Potential multicollinearity found in: " + "; ".join(pair_names) + ".")

    categorical_df = df.select_dtypes(exclude="number")
    if not categorical_df.empty:
        high_card = []
        for col in categorical_df.columns:
            nunique = categorical_df[col].nunique(dropna=True)
            if nunique > max(20, len(df) * 0.2):
                high_card.append(f"{col} ({nunique} unique)")
        if high_card:
            insights.append("High-cardinality categorical columns may need careful encoding: " + ", ".join(high_card[:3]) + ".")

    if not insights:
        insights.append("This dataset looks fairly clean. You can move quickly into visualization and modeling.")

    return insights


def capture_preplify_report(df: pd.DataFrame) -> str:
    return _capture_output(data_report, df)


def capture_preplify_recommendations(df: pd.DataFrame) -> str:
    return _capture_output(recommend_preprocessing, df)


def _capture_output(func, df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = func(df)
    text = buffer.getvalue().strip()
    if text:
        return text
    if result is None:
        return ""
    return str(result)


def report_to_markdown(df: pd.DataFrame) -> str:
    profile = build_dataset_profile(df)
    column_summary = build_column_summary(df)
    insights = generate_smart_insights(df)

    lines = [
        "# Preplify ML Studio Pro Report",
        "",
        "## Dataset Overview",
        f"- Rows: {profile['rows']}",
        f"- Columns: {profile['columns']}",
        f"- Missing values: {profile['missing_values']}",
        f"- Duplicate rows: {profile['duplicate_rows']}",
        f"- Memory usage (MB): {profile['memory_mb']}",
        f"- Data health score: {profile['health_score']}/100",
        "",
        "## Smart Insights",
    ]
    lines.extend([f"- {item}" for item in insights])
    lines.extend(["", "## Column Summary"])

    for _, row in column_summary.iterrows():
        lines.append(
            f"- {row['column']} | dtype={row['dtype']} | missing={row['missing']} | "
            f"missing%={row['missing_%']} | unique={row['unique_values']} | sample={row['sample']}"
        )
    return "\n".join(lines)


def _safe_sample(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "NA"
    value = non_null.iloc[0]
    text = str(value)
    return text[:50] + ("..." if len(text) > 50 else "")
