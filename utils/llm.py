from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Optional SDK import
GROQ_AVAILABLE = False
GROQ_IMPORT_ERROR = None

try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    Groq = None  # type: ignore
    GROQ_IMPORT_ERROR = str(exc)


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com"


def resolve_llm_config(
    provider_choice: str = "Auto (from env)",
    manual_api_key: str | None = None,
    manual_base_url: str | None = None,
    manual_model: str | None = None,
) -> Dict[str, str]:
    """Resolve active LLM config. This app is Groq-only."""
    _ = provider_choice  # Kept for compatibility with existing caller signature.
    return {
        "provider": "groq",
        "api_key": manual_api_key or os.getenv("GROQ_API_KEY", ""),
        "base_url": manual_base_url or os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL),
        "model": manual_model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
    }


def llm_status(config: Dict[str, str] | None = None) -> str:
    if config is None:
        config = resolve_llm_config("Auto (from env)")

    if not GROQ_AVAILABLE:
        return f"Groq SDK not installed. Details: {GROQ_IMPORT_ERROR}"

    api_key = config.get("api_key", "")
    model = config.get("model", DEFAULT_GROQ_MODEL)

    if api_key:
        return f"LLM ready: Groq | model={model}"
    return "Groq selected but no API key found."


def get_client(
    provider: str = "groq",
    api_key: str | None = None,
    base_url: str | None = None,
):
    provider = (provider or "").strip().lower()
    if provider != "groq":
        raise RuntimeError("Unsupported provider. Use Groq.")

    if not GROQ_AVAILABLE or Groq is None:
        raise RuntimeError(f"Groq SDK import failed: {GROQ_IMPORT_ERROR}")

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY.")

    kwargs: Dict[str, Any] = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    else:
        env_base_url = os.getenv("GROQ_BASE_URL")
        if env_base_url:
            kwargs["base_url"] = env_base_url
    return Groq(**kwargs)


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
        text = text[start : end + 1]
    return json.loads(text)


def _chat_text(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    provider: str = "groq",
) -> str:
    provider = provider.lower()
    if provider != "groq":
        raise RuntimeError(f"Unsupported provider in _chat_text: {provider}")

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
    provider: str = "groq",
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    provider = provider.lower()
    if provider != "groq":
        raise RuntimeError("Unsupported provider. Use Groq.")

    if not model_name:
        model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    client = get_client(provider=provider, api_key=api_key, base_url=base_url)
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

    return _chat_text(client, model_name, system_prompt, user_prompt, provider=provider)


def generate_bi_report_spec(
    df: pd.DataFrame,
    user_request: str | None = None,
    prompt: str | None = None,
    provider: str = "groq",
    model_name: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Supports both:
    - old signature style using user_request + model
    - new signature style using prompt + model_name
    """
    provider = provider.lower()
    if provider != "groq":
        raise RuntimeError("Unsupported provider. Use Groq.")

    actual_request = prompt if prompt is not None else user_request
    actual_model = model_name if model_name is not None else model

    if not actual_request:
        actual_request = "Create a business intelligence dashboard for this dataset."

    if not actual_model:
        actual_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    client = get_client(provider=provider, api_key=api_key, base_url=base_url)
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
{actual_request}
"""

    text = _chat_text(client, actual_model, system_prompt, user_prompt, provider=provider)

    try:
        spec = _safe_json_load(text)
    except Exception:
        spec = fallback_bi_report_spec(df, actual_request)

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
