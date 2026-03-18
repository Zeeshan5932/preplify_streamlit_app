from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR


@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, float]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame | None
    confusion: pd.DataFrame | None



def build_model(task: str, model_name: str, params: Dict[str, Any]):
    if task == "classification":
        if model_name == "Logistic Regression":
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=int(params.get("max_iter", 500)),
                solver="lbfgs",
            )
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=None if params.get("max_depth") in (None, 0, "0") else int(params.get("max_depth")),
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=42,
            )
        if model_name == "Gradient Boosting":
            return GradientBoostingClassifier(
                n_estimators=int(params.get("n_estimators", 150)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                random_state=42,
            )
        if model_name == "KNN":
            return KNeighborsClassifier(n_neighbors=int(params.get("n_neighbors", 5)))
        if model_name == "SVM":
            return SVC(C=float(params.get("C", 1.0)), kernel=params.get("kernel", "rbf"), probability=True)
    else:
        if model_name == "Linear Regression":
            return LinearRegression()
        if model_name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=None if params.get("max_depth") in (None, 0, "0") else int(params.get("max_depth")),
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=42,
            )
        if model_name == "Gradient Boosting":
            return GradientBoostingRegressor(
                n_estimators=int(params.get("n_estimators", 150)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                random_state=42,
            )
        if model_name == "KNN":
            return KNeighborsRegressor(n_neighbors=int(params.get("n_neighbors", 5)))
        if model_name == "SVR":
            return SVR(C=float(params.get("C", 1.0)), kernel=params.get("kernel", "rbf"))
    raise ValueError(f"Unsupported model selection: task={task}, model={model_name}")



def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )



def _feature_names(preprocessor: ColumnTransformer, original_columns: list[str]) -> list[str]:
    names: list[str] = []
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = original_columns
    return names



def _importance_df(model: Any, preprocessor: ColumnTransformer, original_columns: list[str]) -> pd.DataFrame | None:
    names = _feature_names(preprocessor, original_columns)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = np.ravel(coef if coef.ndim == 1 else np.mean(np.abs(coef), axis=0))
    else:
        return None

    if len(names) != len(values):
        min_len = min(len(names), len(values))
        names = names[:min_len]
        values = values[:min_len]

    imp = pd.DataFrame({"feature": names, "importance": values})
    imp["importance"] = imp["importance"].abs()
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp



def train_model(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    model_name: str,
    params: Dict[str, Any],
    test_size: float = 0.2,
) -> TrainResult:
    if target_col not in df.columns:
        raise ValueError("Selected target column was not found in the dataframe.")

    clean_df = df.dropna(subset=[target_col]).copy()
    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y if task == "classification" and y.nunique() > 1 else None,
    )

    preprocessor = build_preprocessor(X)
    estimator = build_model(task, model_name, params)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics: Dict[str, float] = {}
    confusion_df = None

    if task == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision_weighted": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
        }
        labels = np.unique(np.concatenate([np.asarray(y_test), np.asarray(preds)]))
        cm = confusion_matrix(y_test, preds, labels=labels)
        confusion_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    else:
        rmse = mean_squared_error(y_test, preds) ** 0.5
        metrics = {
            "r2": round(float(r2_score(y_test, preds)), 4),
            "mae": round(float(mean_absolute_error(y_test, preds)), 4),
            "rmse": round(float(rmse), 4),
        }

    importance = _importance_df(pipe.named_steps["model"], pipe.named_steps["preprocessor"], list(X.columns))
    pred_df = pd.DataFrame({"actual": y_test.reset_index(drop=True), "prediction": pd.Series(preds).reset_index(drop=True)})

    return TrainResult(
        model=pipe,
        metrics=metrics,
        predictions=pred_df,
        feature_importance=importance,
        confusion=confusion_df,
    )



def compare_models(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    test_size: float = 0.2,
) -> pd.DataFrame:
    if task == "classification":
        candidates = [
            ("Logistic Regression", {}),
            ("Random Forest", {"n_estimators": 200}),
            ("Gradient Boosting", {"n_estimators": 150, "learning_rate": 0.1}),
            ("KNN", {"n_neighbors": 5}),
        ]
        score_name = "accuracy"
    else:
        candidates = [
            ("Linear Regression", {}),
            ("Random Forest", {"n_estimators": 200}),
            ("Gradient Boosting", {"n_estimators": 150, "learning_rate": 0.1}),
            ("KNN", {"n_neighbors": 5}),
        ]
        score_name = "r2"

    rows = []
    for name, params in candidates:
        try:
            result = train_model(df, target_col, task, name, params, test_size)
            row = {"model": name, **result.metrics}
            rows.append(row)
        except Exception as exc:
            rows.append({"model": name, "error": str(exc)})

    leaderboard = pd.DataFrame(rows)
    if score_name in leaderboard.columns:
        leaderboard = leaderboard.sort_values(score_name, ascending=False, na_position="last")
    return leaderboard.reset_index(drop=True)



def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric = df.select_dtypes(include="number").copy()
    if numeric.empty:
        raise ValueError("Anomaly detection requires at least one numeric column.")

    numeric = numeric.fillna(numeric.median(numeric_only=True))
    detector = IsolationForest(contamination=contamination, random_state=42)
    preds = detector.fit_predict(numeric)
    scores = detector.decision_function(numeric)

    output = df.copy()
    output["anomaly_flag"] = np.where(preds == -1, "anomaly", "normal")
    output["anomaly_score"] = scores

    summary = output["anomaly_flag"].value_counts().rename_axis("class").reset_index(name="count")
    return output, summary
