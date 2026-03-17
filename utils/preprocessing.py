from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import List, Tuple

import pandas as pd
from preplify import PreplifyPipeline, auto_prep


def preprocess_with_preplify(
    df: pd.DataFrame,
    use_auto_prep: bool,
    missing_strategy: str,
    encoding: str,
    scaling: str,
    outlier_method: str,
    feature_engineering: bool,
    fill_value: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run Preplify preprocessing and capture a readable log for the UI."""
    logs: List[str] = []
    work_df = df.copy()

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        if use_auto_prep:
            processed = auto_prep(work_df)
            logs.append("Used preplify.auto_prep(df)")
        else:
            pipeline_kwargs = {
                "missing_strategy": missing_strategy,
                "encoding": encoding,
                "scaling": scaling,
                "feature_engineering": feature_engineering,
                "outlier_method": None if outlier_method == "none" else outlier_method,
            }
            if missing_strategy == "constant":
                parsed_fill_value = _coerce_fill_value(fill_value)
                pipeline_kwargs["fill_value"] = parsed_fill_value
                logs.append(f"Constant fill value: {parsed_fill_value!r}")

            pipe = PreplifyPipeline(**pipeline_kwargs)
            processed = pipe.fit_transform(work_df)
            logs.append(f"Used PreplifyPipeline with {pipeline_kwargs}")

    stdout_text = buffer.getvalue().strip()
    if stdout_text:
        logs.append("Captured stdout from Preplify:")
        logs.append(stdout_text)

    if not isinstance(processed, pd.DataFrame):
        processed = pd.DataFrame(processed)
        logs.append("Preplify returned a non-DataFrame object; converted it to DataFrame.")

    return processed, logs


def _coerce_fill_value(raw: str):
    raw = raw.strip()
    if raw == "":
        return ""
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
