from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

PREPLIFY_AVAILABLE = False
PREPLIFY_IMPORT_ERROR = None

try:
    from preplify import (  # type: ignore
        PreplifyPipeline,
        auto_prep,
        data_report,
        recommend_preprocessing,
    )

    PREPLIFY_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import fallback for environments without preplify
    PREPLIFY_IMPORT_ERROR = str(exc)
    PreplifyPipeline = None  # type: ignore
    auto_prep = None  # type: ignore
    data_report = None  # type: ignore
    recommend_preprocessing = None  # type: ignore


DEFAULT_RECOMMENDATIONS = [
    "Review missing values before modeling.",
    "Inspect outliers in numeric columns.",
    "Encode categorical columns before training models.",
    "Scale numeric columns for distance-based models.",
    "Check target leakage before feature selection.",
]


def status_message() -> str:
    if PREPLIFY_AVAILABLE:
        return "Preplify imported successfully."
    return (
        "Preplify is not installed or failed to import in this environment. "
        f"Fallback preprocessing will be used. Details: {PREPLIFY_IMPORT_ERROR}"
    )



def _normalize_report_output(output: Any) -> Dict[str, Any]:
    if output is None:
        return {}
    if isinstance(output, dict):
        return output
    if isinstance(output, pd.DataFrame):
        return {
            "type": "dataframe",
            "shape": list(output.shape),
            "records": output.head(25).to_dict(orient="records"),
        }
    if isinstance(output, pd.Series):
        return output.to_dict()
    return {"raw_output": str(output)}



def get_report(df: pd.DataFrame) -> Dict[str, Any]:
    if PREPLIFY_AVAILABLE and data_report is not None:
        try:
            output = data_report(df.copy())
            return _normalize_report_output(output)
        except Exception as exc:
            return {"warning": f"Preplify data_report failed: {exc}"}
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
    }



def get_recommendations(df: pd.DataFrame) -> List[str]:
    if PREPLIFY_AVAILABLE and recommend_preprocessing is not None:
        try:
            output = recommend_preprocessing(df.copy())
            if output is None:
                return DEFAULT_RECOMMENDATIONS
            if isinstance(output, list):
                return [str(item) for item in output]
            if isinstance(output, pd.DataFrame):
                return output.astype(str).apply(" | ".join, axis=1).tolist()
            if isinstance(output, dict):
                recs: List[str] = []
                for key, value in output.items():
                    recs.append(f"{key}: {value}")
                return recs or DEFAULT_RECOMMENDATIONS
            return [str(output)]
        except Exception as exc:
            return [f"Preplify recommendation step failed: {exc}"] + DEFAULT_RECOMMENDATIONS
    return DEFAULT_RECOMMENDATIONS



def apply_auto_prep(df: pd.DataFrame) -> pd.DataFrame:
    if PREPLIFY_AVAILABLE and auto_prep is not None:
        try:
            return auto_prep(df.copy())
        except Exception:
            pass

    out = df.copy()
    out = out.drop_duplicates().reset_index(drop=True)

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].median())
        else:
            if out[col].isna().any():
                mode = out[col].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else "missing"
                out[col] = out[col].fillna(fill)

    categorical_cols = out.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if categorical_cols:
        out = pd.get_dummies(out, columns=categorical_cols, drop_first=False)

    return out



def apply_custom_pipeline(
    df: pd.DataFrame,
    missing_strategy: str = "median",
    encoding: str = "onehot",
    scaling: str = "standard",
    outlier_method: str | None = None,
    feature_engineering: bool = False,
) -> pd.DataFrame:
    if PREPLIFY_AVAILABLE and PreplifyPipeline is not None:
        try:
            pipeline = PreplifyPipeline(
                missing_strategy=missing_strategy,
                encoding=encoding,
                scaling=scaling,
                outlier_method=outlier_method,
                feature_engineering=feature_engineering,
            )
            return pipeline.fit_transform(df.copy())
        except Exception:
            pass

    out = apply_auto_prep(df)

    # Light scaling fallback on numeric features.
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    if scaling in {"standard", "minmax", "robust"} and numeric_cols:
        nums = out[numeric_cols].copy()
        if scaling == "standard":
            std = nums.std(ddof=0).replace(0, 1)
            out[numeric_cols] = (nums - nums.mean()) / std
        elif scaling == "minmax":
            denom = (nums.max() - nums.min()).replace(0, 1)
            out[numeric_cols] = (nums - nums.min()) / denom
        elif scaling == "robust":
            iqr = (nums.quantile(0.75) - nums.quantile(0.25)).replace(0, 1)
            out[numeric_cols] = (nums - nums.median()) / iqr

    return out
