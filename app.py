import io
import json
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st

from utils.modeling import (
    build_model,
    classification_metrics,
    get_default_problem_type,
    get_model_options,
    regression_metrics,
)
from utils.preprocessing import preprocess_with_preplify
from utils.reporting import (
    build_basic_report,
    capture_preplify_recommendations,
    capture_preplify_report,
    report_to_text,
)
from utils.visuals import render_visual_section


st.set_page_config(
    page_title="Preplify ML Studio",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Preplify ML Studio")
st.caption(
    "Upload a CSV, generate a report, visualize your data, preprocess it with Preplify, and train a machine learning model from the browser."
)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def show_dataframe_preview(df: pd.DataFrame, title: str) -> None:
    st.subheader(title)
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")


with st.sidebar:
    st.header("1) Data source")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    st.markdown("---")
    st.header("2) Navigation")
    section = st.radio(
        "Open section",
        ["Overview", "Report", "Visualize", "Preprocess", "Train Model"],
        label_visibility="collapsed",
    )

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to start.")
    st.stop()

try:
    df = load_uploaded_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()

if df.empty:
    st.warning("The uploaded file is empty.")
    st.stop()

if section == "Overview":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing values", int(df.isna().sum().sum()))
    col4.metric("Duplicate rows", int(df.duplicated().sum()))

    show_dataframe_preview(df, "Raw data preview")

    st.subheader("Column types")
    dtypes_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing": df.isna().sum().values,
            "unique_values": [df[col].nunique(dropna=False) for col in df.columns],
        }
    )
    st.dataframe(dtypes_df, use_container_width=True)

elif section == "Report":
    st.subheader("Dataset report")
    basic_report = build_basic_report(df)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Built-in summary")
        st.json(basic_report)
    with right:
        st.markdown("#### Download summary")
        report_text = report_to_text(basic_report)
        st.download_button(
            "Download report.txt",
            data=report_text,
            file_name="preplify_report.txt",
            mime="text/plain",
        )

    st.markdown("#### Preplify data report")
    preplify_report_output = capture_preplify_report(df)
    if preplify_report_output:
        st.code(preplify_report_output)
    else:
        st.info("No printable output was returned by data_report().")

    st.markdown("#### Preplify recommendations")
    preplify_reco_output = capture_preplify_recommendations(df)
    if preplify_reco_output:
        st.code(preplify_reco_output)
    else:
        st.info("No printable output was returned by recommend_preprocessing().")

elif section == "Visualize":
    render_visual_section(df)

elif section == "Preprocess":
    st.subheader("Preprocess your data with Preplify")

    with st.form("preprocess_form"):
        c1, c2, c3 = st.columns(3)
        missing_strategy = c1.selectbox(
            "Missing strategy",
            ["mean", "median", "mode", "drop", "constant"],
            index=1,
        )
        encoding = c2.selectbox("Encoding", ["onehot", "label"], index=0)
        scaling = c3.selectbox("Scaling", ["standard", "minmax", "robust"], index=0)

        c4, c5, c6 = st.columns(3)
        outlier_method = c4.selectbox("Outlier method", ["none", "iqr", "zscore"], index=0)
        feature_engineering = c5.checkbox("Enable feature engineering", value=False)
        fill_value = c6.text_input("Constant fill value", value="0")

        preprocess_mode = st.radio(
            "Preprocessing mode",
            ["Custom pipeline", "One-click auto_prep"],
            horizontal=True,
        )

        submitted = st.form_submit_button("Run preprocessing")

    if submitted:
        with st.spinner("Running Preplify preprocessing..."):
            try:
                processed_df, log_messages = preprocess_with_preplify(
                    df=df,
                    use_auto_prep=preprocess_mode == "One-click auto_prep",
                    missing_strategy=missing_strategy,
                    encoding=encoding,
                    scaling=scaling,
                    outlier_method=outlier_method,
                    feature_engineering=feature_engineering,
                    fill_value=fill_value,
                )
                st.session_state["processed_df"] = processed_df
                st.session_state["preprocess_log"] = log_messages
            except Exception as exc:
                st.error(f"Preprocessing failed: {exc}")

    processed_df = st.session_state.get("processed_df")
    if processed_df is not None:
        st.success("Preprocessing finished.")
        show_dataframe_preview(processed_df, "Processed data preview")
        st.markdown("#### Preprocessing log")
        st.code("\n".join(st.session_state.get("preprocess_log", [])) or "No extra log available.")
        st.download_button(
            "Download processed_data.csv",
            data=processed_df.to_csv(index=False),
            file_name="processed_data.csv",
            mime="text/csv",
        )
    else:
        st.info("Choose your preprocessing settings and click 'Run preprocessing'.")

elif section == "Train Model":
    st.subheader("Train your own model")

    source_choice = st.radio(
        "Choose data source for training",
        ["Raw uploaded data", "Processed data from Preprocess tab"],
        horizontal=True,
    )

    train_df = df if source_choice == "Raw uploaded data" else st.session_state.get("processed_df")
    if train_df is None:
        st.warning("No processed data found yet. Run preprocessing first or switch to raw data.")
        st.stop()

    show_dataframe_preview(train_df, "Training dataset preview")

    target_col = st.selectbox("Target column", train_df.columns)
    default_problem_type = get_default_problem_type(train_df[target_col])
    problem_type = st.radio(
        "Problem type",
        ["classification", "regression"],
        index=0 if default_problem_type == "classification" else 1,
        horizontal=True,
    )

    model_choices = get_model_options(problem_type)
    model_name = st.selectbox("Model", list(model_choices.keys()))

    st.markdown("#### Model parameters")
    model_key = model_choices[model_name]
    param_values = {}

    if model_key == "logistic_regression":
        c1, c2 = st.columns(2)
        param_values["C"] = c1.slider("C", 0.01, 10.0, 1.0)
        param_values["max_iter"] = c2.slider("max_iter", 100, 2000, 500, step=100)
    elif model_key == "random_forest_classifier":
        c1, c2, c3 = st.columns(3)
        param_values["n_estimators"] = c1.slider("n_estimators", 50, 500, 200, step=50)
        param_values["max_depth"] = c2.selectbox("max_depth", [None, 3, 5, 10, 20], index=0)
        param_values["min_samples_split"] = c3.slider("min_samples_split", 2, 10, 2)
    elif model_key == "knn_classifier":
        c1, c2 = st.columns(2)
        param_values["n_neighbors"] = c1.slider("n_neighbors", 1, 25, 5)
        param_values["weights"] = c2.selectbox("weights", ["uniform", "distance"])
    elif model_key == "svc":
        c1, c2 = st.columns(2)
        param_values["C"] = c1.slider("C", 0.01, 10.0, 1.0)
        param_values["kernel"] = c2.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
    elif model_key == "linear_regression":
        st.info("LinearRegression uses default parameters in this template.")
    elif model_key == "ridge":
        param_values["alpha"] = st.slider("alpha", 0.01, 10.0, 1.0)
    elif model_key == "random_forest_regressor":
        c1, c2, c3 = st.columns(3)
        param_values["n_estimators"] = c1.slider("n_estimators", 50, 500, 200, step=50)
        param_values["max_depth"] = c2.selectbox("max_depth", [None, 3, 5, 10, 20], index=0)
        param_values["min_samples_split"] = c3.slider("min_samples_split", 2, 10, 2)
    elif model_key == "knn_regressor":
        c1, c2 = st.columns(2)
        param_values["n_neighbors"] = c1.slider("n_neighbors", 1, 25, 5)
        param_values["weights"] = c2.selectbox("weights", ["uniform", "distance"])
    elif model_key == "svr":
        c1, c2 = st.columns(2)
        param_values["C"] = c1.slider("C", 0.01, 10.0, 1.0)
        param_values["kernel"] = c2.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])

    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test size", 0.1, 0.4, 0.2, step=0.05)
    random_state = c2.number_input("Random state", min_value=0, value=42, step=1)
    preprocess_before_training = c3.checkbox(
        "Apply simple auto_prep before training",
        value=False,
        help="Useful when training directly from raw data.",
    )

    if st.button("Train model"):
        try:
            work_df = train_df.copy()
            y = work_df[target_col]
            X = work_df.drop(columns=[target_col])

            if preprocess_before_training:
                X, prep_logs = preprocess_with_preplify(
                    df=X,
                    use_auto_prep=True,
                    missing_strategy="median",
                    encoding="onehot",
                    scaling="standard",
                    outlier_method="none",
                    feature_engineering=False,
                    fill_value="0",
                )
                y = y.loc[X.index]
                st.info("Applied auto_prep() before training.")
                st.code("\n".join(prep_logs))

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=int(random_state),
                stratify=y if problem_type == "classification" and y.nunique() > 1 else None,
            )

            model = build_model(problem_type, model_key, param_values, random_state=int(random_state))
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.markdown("#### Training results")
            if problem_type == "classification":
                metrics = classification_metrics(y_test, preds)
            else:
                metrics = regression_metrics(y_test, preds)

            mcol1, mcol2, mcol3 = st.columns(3)
            metric_items = list(metrics.items())
            for idx, (name, value) in enumerate(metric_items[:3]):
                [mcol1, mcol2, mcol3][idx].metric(name, f"{value:.4f}")

            if len(metric_items) > 3:
                extra_metrics = {k: round(v, 4) for k, v in metric_items[3:]}
                st.json(extra_metrics)

            preview_df = pd.DataFrame({"actual": y_test.values, "prediction": preds})
            st.markdown("#### Predictions preview")
            st.dataframe(preview_df.head(20), use_container_width=True)

            payload = {
                "problem_type": problem_type,
                "model": model_name,
                "params": param_values,
                "metrics": {k: float(v) for k, v in metrics.items()},
            }
            st.download_button(
                "Download metrics.json",
                data=json.dumps(payload, indent=2),
                file_name="model_metrics.json",
                mime="application/json",
            )

        except Exception as exc:
            st.error(f"Training failed: {exc}")
