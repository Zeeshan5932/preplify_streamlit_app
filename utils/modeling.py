from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR


def get_default_problem_type(target: pd.Series) -> str:
    if str(target.dtype) == "object" or str(target.dtype).startswith("category"):
        return "classification"
    if target.nunique(dropna=True) <= 20:
        return "classification"
    return "regression"


def get_model_options(problem_type: str) -> Dict[str, str]:
    if problem_type == "classification":
        return {
            "Logistic Regression": "logistic_regression",
            "Random Forest Classifier": "random_forest_classifier",
            "KNN Classifier": "knn_classifier",
            "Support Vector Classifier": "svc",
        }
    return {
        "Linear Regression": "linear_regression",
        "Ridge Regression": "ridge",
        "Random Forest Regressor": "random_forest_regressor",
        "KNN Regressor": "knn_regressor",
        "Support Vector Regressor": "svr",
    }


def build_model(problem_type: str, model_key: str, params: dict, random_state: int = 42):
    if problem_type == "classification":
        if model_key == "logistic_regression":
            return LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 500),
                random_state=random_state,
            )
        if model_key == "random_forest_classifier":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth"),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=random_state,
            )
        if model_key == "knn_classifier":
            return KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "uniform"),
            )
        if model_key == "svc":
            return SVC(
                C=params.get("C", 1.0),
                kernel=params.get("kernel", "rbf"),
            )
    else:
        if model_key == "linear_regression":
            return LinearRegression()
        if model_key == "ridge":
            return Ridge(alpha=params.get("alpha", 1.0), random_state=random_state)
        if model_key == "random_forest_regressor":
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth"),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=random_state,
            )
        if model_key == "knn_regressor":
            return KNeighborsRegressor(
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "uniform"),
            )
        if model_key == "svr":
            return SVR(
                C=params.get("C", 1.0),
                kernel=params.get("kernel", "rbf"),
            )

    raise ValueError(f"Unsupported model: {model_key}")


def classification_metrics(y_true, y_pred) -> dict:
    average = "weighted"
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def regression_metrics(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "r2_score": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
    }
