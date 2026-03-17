from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
            "Random Forest": "random_forest_classifier",
            "Gradient Boosting": "gradient_boosting_classifier",
            "KNN": "knn_classifier",
            "SVC": "svc",
        }
    return {
        "Linear Regression": "linear_regression",
        "Ridge Regression": "ridge",
        "Random Forest": "random_forest_regressor",
        "Gradient Boosting": "gradient_boosting_regressor",
        "KNN": "knn_regressor",
        "SVR": "svr",
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
        if model_key == "gradient_boosting_classifier":
            return GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 150),
                learning_rate=params.get("learning_rate", 0.1),
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
                probability=True,
                random_state=random_state,
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
        if model_key == "gradient_boosting_regressor":
            return GradientBoostingRegressor(
                n_estimators=params.get("n_estimators", 150),
                learning_rate=params.get("learning_rate", 0.1),
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


def prepare_training_matrices(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    work_df = df.copy()
    work_df = work_df.dropna(subset=[target_col])
    y = work_df[target_col]
    X = work_df.drop(columns=[target_col])

    X = pd.get_dummies(X, dummy_na=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=work_df.index)

    return X, y


def fit_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    model_key: str,
    params: dict,
    test_size: float,
    random_state: int,
    use_cv: bool,
    cv_folds: int,
):
    X, y = prepare_training_matrices(df, target_col)

    label_encoder = None
    y_for_model = y.copy()
    if problem_type == "classification" and (str(y.dtype) == "object" or str(y.dtype).startswith("category")):
        label_encoder = LabelEncoder()
        y_for_model = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)

    stratify_target = y_for_model if problem_type == "classification" and pd.Series(y_for_model).nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_for_model,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    model = build_model(problem_type, model_key, params, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if problem_type == "classification":
        metrics = classification_metrics(y_test, preds)
        probas = _predict_proba_or_score(model, X_test)
        if probas is not None and len(pd.Series(y_test).unique()) == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, probas)
        details = {
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "classes": _classes_for_display(label_encoder, y_for_model),
        }
    else:
        metrics = regression_metrics(y_test, preds)
        probas = None
        details = {}

    cv_scores = None
    if use_cv:
        scoring = "accuracy" if problem_type == "classification" else "r2"
        cv_scores = cross_val_score(model, X, y_for_model, cv=cv_folds, scoring=scoring)

    feature_importance = get_feature_importance(model, X.columns.tolist())

    result = {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": preds,
        "metrics": metrics,
        "details": details,
        "feature_importance": feature_importance,
        "cv_scores": cv_scores,
        "encoded": label_encoder is not None,
        "label_encoder": label_encoder,
    }
    return result


def compare_models(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    X, y = prepare_training_matrices(df, target_col)
    if problem_type == "classification" and (str(y.dtype) == "object" or str(y.dtype).startswith("category")):
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)

    stratify_target = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    rows: List[dict] = []
    for model_name, model_key in get_model_options(problem_type).items():
        model = build_model(problem_type, model_key, params={}, random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if problem_type == "classification":
            metrics = classification_metrics(y_test, preds)
            score = metrics["f1_score"]
        else:
            metrics = regression_metrics(y_test, preds)
            score = metrics["r2_score"]
        row = {"model": model_name, **{k: round(float(v), 4) for k, v in metrics.items()}, "score": round(float(score), 4)}
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return leaderboard


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


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

    if values is None:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({"feature": feature_names, "importance": values})
    return importance_df.sort_values("importance", ascending=False).head(20).reset_index(drop=True)


def _predict_proba_or_score(model, X_test):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        if np.asarray(scores).ndim == 1:
            return scores
    return None


def _classes_for_display(label_encoder, y):
    if label_encoder is not None:
        return label_encoder.classes_.tolist()
    return sorted(pd.Series(y).astype(str).unique().tolist())
