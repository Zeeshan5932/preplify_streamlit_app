from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any, Dict

import pandas as pd
from preplify import data_report, recommend_preprocessing



def build_basic_report(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "column_summary": [
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "unique_values": int(df[col].nunique(dropna=False)),
            }
            for col in df.columns
        ],
    }



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



def report_to_text(report: Dict[str, Any]) -> str:
    lines = ["PREPLIFY DATA REPORT", "=" * 40, ""]
    shape = report["shape"]
    lines.append(f"Rows: {shape['rows']}")
    lines.append(f"Columns: {shape['columns']}")
    lines.append(f"Missing values: {report['missing_values']}")
    lines.append(f"Duplicate rows: {report['duplicate_rows']}")
    lines.append("")
    lines.append("NUMERIC COLUMNS")
    lines.append(", ".join(report["numeric_columns"]) or "None")
    lines.append("")
    lines.append("CATEGORICAL COLUMNS")
    lines.append(", ".join(report["categorical_columns"]) or "None")
    lines.append("")
    lines.append("COLUMN SUMMARY")
    for item in report["column_summary"]:
        lines.append(
            f"- {item['column']} | dtype={item['dtype']} | missing={item['missing']} | unique={item['unique_values']}"
        )
    return "\n".join(lines)
