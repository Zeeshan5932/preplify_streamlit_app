from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

GROQ_AVAILABLE = False
GROQ_IMPORT_ERROR = None

try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    Groq = None  # type: ignore
    GROQ_IMPORT_ERROR = str(exc)


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def llm_status() -> str:
    if GROQ_AVAILABLE:
        return "Groq LLM module ready. Add GROQ_API_KEY to enable AI chat and BI report generation."
    return f"Groq SDK not installed. Details: {GROQ_IMPORT_ERROR}"


def get_client(api_key: str | None = None):
    if not GROQ_AVAILABLE or Groq is None:
        raise RuntimeError(f"Groq SDK import failed: {GROQ_IMPORT_ERROR}")

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing API key. Set GROQ_API_KEY or enter it in the app.")

    return Groq(api_key=key)


def dataset_context(df: pd.DataFrame, max_cols: int = 25) -> Dict[str, Any]:
    cols = list(df.columns[:max_cols])
    trimmed = df[cols].copy() if cols else df.copy()
    numeric_cols = trimmed.select_dtypes(include="number").columns.tolist()
    categorical_cols = trimmed.select_dtypes(exclude="number").columns.tolist()
    sample_rows = trimmed.head(8).astype(str).to_dict(orient="records")

    context = {
        "shape": list(df.shape),
        "visible_columns": cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "dtypes": {col: str(dtype) for col, dtype in trimmed.dtypes.items()},
        "missing": trimmed.isna().sum().to_dict(),
        "numeric_summary": trimmed[numeric_cols].describe().round(3).to_dict() if numeric_cols else {},
        "top_categories": {
            col: trimmed[col].astype(str).value_counts().head(5).to_dict()
            for col in categorical_cols[:8]
        },
        "sample_rows": sample_rows,
    }
    return context


def _safe_json_load(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def _chat_text(client, model: str, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def ask_dataset_question(
    df: pd.DataFrame,
    question: str,
    model: str = DEFAULT_GROQ_MODEL,
    api_key: str | None = None,
) -> str:
    client = get_client(api_key=api_key)
    context = dataset_context(df)

    system_prompt = (
        "You are an AI data analyst inside a Streamlit app. "
        "Answer ONLY using the dataset context provided. "
        "If the answer is not available in the context, say so clearly. "
        "Be concise, practical, and helpful."
    )

    user_prompt = f"""
DATASET CONTEXT:
{json.dumps(context, indent=2)}

USER QUESTION:
{question}
"""
    return _chat_text(client, model, system_prompt, user_prompt)


def generate_bi_report_spec(
    df: pd.DataFrame,
    user_request: str,
    model: str = DEFAULT_GROQ_MODEL,
    api_key: str | None = None,
) -> Tuple[Dict[str, Any], str]:
    client = get_client(api_key=api_key)
    context = dataset_context(df)

    allowed_chart_types = [
        "bar",
        "line",
        "area",
        "scatter",
        "histogram",
        "box",
        "pie",
        "heatmap",
        "3d_scatter",
    ]

    system_prompt = (
        "You generate BI dashboard specs for a Streamlit app. "
        "The dashboard should feel like Tableau or Power BI: executive summary, KPI cards, "
        "clear chart titles, and concise insights. "
        "Use ONLY columns that exist in the dataset context. "
        "Return ONLY valid JSON."
    )

    user_prompt = f"""
JSON schema:
{{
  "title": "string",
  "subtitle": "string",
  "theme": "light or dark",
  "summary": "short executive summary",
  "kpis": [
    {{"label": "string", "type": "row_count|unique_count|sum|avg|missing_count", "column": "optional column name"}}
  ],
  "insights": ["string", "string"],
  "charts": [
    {{
      "chart_type": "one of {allowed_chart_types}",
      "title": "string",
      "x": "optional column",
      "y": "optional column",
      "z": "optional column",
      "color": "optional column",
      "aggregation": "none|sum|mean|count"
    }}
  ]
}}

DATASET CONTEXT:
{json.dumps(context, indent=2)}

USER REQUEST:
{user_request}
"""

    text = _chat_text(client, model, system_prompt, user_prompt)

    try:
        spec = _safe_json_load(text)
    except Exception:
        spec = fallback_bi_report_spec(df, user_request)

    return spec, text


def fallback_bi_report_spec(df: pd.DataFrame, user_request: str = "") -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    charts: List[Dict[str, Any]] = []

    if categorical_cols and numeric_cols:
        charts.append(
            {
                "chart_type": "bar",
                "title": f"Average {numeric_cols[0]} by {categorical_cols[0]}",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "color": None,
                "aggregation": "mean",
            }
        )

    if len(numeric_cols) >= 2:
        charts.append(
            {
                "chart_type": "scatter",
                "title": f"{numeric_cols[0]} vs {numeric_cols[1]}",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "color": categorical_cols[0] if categorical_cols else None,
                "aggregation": "none",
            }
        )

    if len(numeric_cols) >= 3:
        charts.append(
            {
                "chart_type": "3d_scatter",
                "title": f"3D view of {numeric_cols[0]}, {numeric_cols[1]}, {numeric_cols[2]}",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "z": numeric_cols[2],
                "color": categorical_cols[0] if categorical_cols else None,
                "aggregation": "none",
            }
        )

    return {
        "title": "AI BI Dashboard",
        "subtitle": user_request or "Auto-generated business intelligence report",
        "theme": "light",
        "summary": "This dashboard summarizes the uploaded dataset with KPIs, chart highlights, and exploratory visuals.",
        "kpis": [
            {"label": "Rows", "type": "row_count", "column": None},
            {"label": "Columns", "type": "unique_count", "column": "__columns__"},
            {"label": "Missing Cells", "type": "missing_count", "column": None},
        ],
        "insights": [
            "Use KPI cards for the first screen to mimic a BI dashboard.",
            "Prioritize category-vs-metric charts and one exploratory scatter chart.",
            "Use a 3D view when you have at least three numeric columns.",
        ],
        "charts": charts,
    }