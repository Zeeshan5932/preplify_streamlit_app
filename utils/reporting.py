from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd



def infer_task_type(series: pd.Series) -> str:
    if series.dtype == object or str(series.dtype).startswith("category") or series.dtype == bool:
        return "classification"
    unique = series.nunique(dropna=True)
    total = max(len(series), 1)
    ratio = unique / total
    if unique <= 20 or ratio < 0.05:
        return "classification"
    return "regression"



def basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }



def health_score(df: pd.DataFrame) -> Dict[str, Any]:
    rows, cols = df.shape
    total_cells = max(rows * max(cols, 1), 1)
    missing_ratio = float(df.isna().sum().sum() / total_cells)
    duplicate_ratio = float(df.duplicated().mean()) if rows > 0 else 0.0

    numeric = df.select_dtypes(include="number")
    outlier_ratio = 0.0
    if not numeric.empty:
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = ((numeric < lower) | (numeric > upper)).fillna(False)
        outlier_ratio = float(mask.mean().mean()) if mask.size else 0.0

    raw_score = 100 - (missing_ratio * 45 + duplicate_ratio * 20 + outlier_ratio * 35) * 100
    score = int(max(0, min(100, round(raw_score))))

    if score >= 85:
        level = "Excellent"
    elif score >= 70:
        level = "Good"
    elif score >= 50:
        level = "Needs attention"
    else:
        level = "Poor"

    return {
        "score": score,
        "level": level,
        "missing_ratio": round(missing_ratio, 4),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "outlier_ratio": round(outlier_ratio, 4),
    }



def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "unique": int(series.nunique(dropna=True)),
                "sample": str(series.dropna().iloc[0]) if series.dropna().shape[0] else "",
            }
        )
    return pd.DataFrame(rows)



def build_report_payload(
    df: pd.DataFrame,
    preplify_report: Dict[str, Any] | None = None,
    recommendations: List[str] | None = None,
) -> Dict[str, Any]:
    payload = {
        "profile": basic_profile(df),
        "health": health_score(df),
        "top_missing_columns": df.isna().sum().sort_values(ascending=False).head(10).to_dict(),
        "column_summary": column_summary(df).to_dict(orient="records"),
        "preplify_report": preplify_report or {},
        "recommendations": recommendations or [],
    }
    return payload



def report_markdown(payload: Dict[str, Any]) -> str:
    profile = payload.get("profile", {})
    health = payload.get("health", {})
    recs = payload.get("recommendations", [])
    lines = [
        "# Smart Dataset Report",
        "",
        "## Dataset Snapshot",
        f"- Rows: {profile.get('rows', 0)}",
        f"- Columns: {profile.get('columns', 0)}",
        f"- Numeric columns: {len(profile.get('numeric_columns', []))}",
        f"- Categorical columns: {len(profile.get('categorical_columns', []))}",
        f"- Missing cells: {profile.get('missing_total', 0)}",
        f"- Duplicate rows: {profile.get('duplicate_rows', 0)}",
        f"- Memory usage (MB): {profile.get('memory_mb', 0)}",
        "",
        "## Data Health",
        f"- Score: {health.get('score', 0)}/100",
        f"- Level: {health.get('level', 'Unknown')}",
        f"- Missing ratio: {health.get('missing_ratio', 0)}",
        f"- Duplicate ratio: {health.get('duplicate_ratio', 0)}",
        f"- Outlier ratio: {health.get('outlier_ratio', 0)}",
        "",
        "## Recommended Next Steps",
    ]
    if recs:
        lines.extend([f"- {item}" for item in recs])
    else:
        lines.append("- No recommendations were generated.")
    return "\n".join(lines)



def report_html(payload: Dict[str, Any]) -> str:
    markdown = report_markdown(payload).replace("\n", "<br>")
    pretty_json = json.dumps(payload, indent=2)
    return f"""
    <html>
    <head>
        <title>Smart Dataset Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
            .card {{ border: 1px solid #d1d5db; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
            pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 14px; border-radius: 10px; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Smart Dataset Report</h1>
            <div>{markdown}</div>
        </div>
        <div class="card">
            <h2>JSON Payload</h2>
            <pre>{pretty_json}</pre>
        </div>
    </body>
    </html>
    """
