from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.modeling import (
    compare_models,
    fit_and_evaluate,
    get_default_problem_type,
    get_model_options,
)
from utils.preprocessing import preprocess_with_preplify
from utils.reporting import (
    build_column_summary,
    build_dataset_profile,
    capture_preplify_recommendations,
    capture_preplify_report,
    generate_smart_insights,
    report_to_markdown,
)
from utils.visuals import render_visual_lab


st.set_page_config(page_title="Preplify ML Studio Pro", page_icon="🚀", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(58,123,213,0.10), rgba(0,210,255,0.10));
        border: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 1rem;
    }
    .small-note {color: #7a7a7a; font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.35rem;">🚀 Preplify ML Studio Pro</h1>
        <div class="small-note">
            Upload a CSV, generate smart reports, explore 2D/3D visuals, preprocess with Preplify,
            compare models, tune parameters, detect anomalies, and export results.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def current_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    processed_df = st.session_state.get("processed_df")
    return processed_df if processed_df is not None else raw_df


def show_top_metrics(df: pd.DataFrame) -> None:
    profile = build_dataset_profile(df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{profile['rows']:,}")
    c2.metric("Columns", profile["columns"])
    c3.metric("Missing", profile["missing_values"])
    c4.metric("Duplicates", profile["duplicate_rows"])
    c5.metric("Memory (MB)", profile["memory_mb"])
    c6.metric("Health score", f"{profile['health_score']}/100")


with st.sidebar:
    st.header("Data source")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    st.markdown("---")
    st.header("Sections")
    section = st.radio(
        "Open",
        ["Overview", "Smart Report", "Visual Lab", "Preprocess Studio", "Model Lab"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Tip: Run preprocessing first if your raw data is very messy.")

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to start building your analytics app.")
    st.stop()

try:
    raw_df = load_uploaded_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()

if raw_df.empty:
    st.warning("The uploaded file is empty.")
    st.stop()

if section == "Overview":
    show_top_metrics(raw_df)

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Raw data preview")
        st.dataframe(raw_df.head(25), use_container_width=True)
    with right:
        st.subheader("Quick facts")
        profile = build_dataset_profile(raw_df)
        st.json(profile)

    st.subheader("Column audit")
    st.dataframe(build_column_summary(raw_df), use_container_width=True)

    st.subheader("Instant insights")
    for insight in generate_smart_insights(raw_df):
        st.info(insight)

    st.download_button(
        "Download overview_report.md",
        data=report_to_markdown(raw_df),
        file_name="overview_report.md",
        mime="text/markdown",
    )

elif section == "Smart Report":
    show_top_metrics(raw_df)

    tab1, tab2, tab3 = st.tabs(["Smart summary", "Preplify report", "Preplify recommendations"])

    with tab1:
        st.subheader("Auto-generated dataset story")
        for insight in generate_smart_insights(raw_df):
            st.write(f"- {insight}")

        summary_df = build_column_summary(raw_df)
        st.dataframe(summary_df, use_container_width=True)
        st.download_button(
            "Download smart_report.md",
            data=report_to_markdown(raw_df),
            file_name="smart_report.md",
            mime="text/markdown",
        )

    with tab2:
        output = capture_preplify_report(raw_df)
        if output:
            st.code(output)
        else:
            st.info("No printable output was returned by Preplify's data_report().")

    with tab3:
        output = capture_preplify_recommendations(raw_df)
        if output:
            st.code(output)
        else:
            st.info("No printable output was returned by Preplify's recommend_preprocessing().")

elif section == "Visual Lab":
    data_source = st.radio(
        "Choose data source",
        ["Raw uploaded data", "Processed data"],
        horizontal=True,
    )
    df_for_vis = raw_df if data_source == "Raw uploaded data" else current_data(raw_df)

    if data_source == "Processed data" and st.session_state.get("processed_df") is None:
        st.warning("No processed data found yet, so the raw data is being shown.")
        df_for_vis = raw_df

    show_top_metrics(df_for_vis)
    render_visual_lab(df_for_vis)

elif section == "Preprocess Studio":
    st.subheader("Preprocess your data with Preplify")

    with st.form("preprocess_form"):
        c1, c2, c3 = st.columns(3)
        missing_strategy = c1.selectbox("Missing strategy", ["mean", "median", "mode", "drop", "constant"], index=1)
        encoding = c2.selectbox("Encoding", ["onehot", "label"], index=0)
        scaling = c3.selectbox("Scaling", ["standard", "minmax", "robust"], index=0)

        c4, c5, c6 = st.columns(3)
        outlier_method = c4.selectbox("Outlier method", ["none", "iqr", "zscore"], index=0)
        feature_engineering = c5.checkbox("Enable feature engineering", value=True)
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
                    df=raw_df,
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
                st.success("Preprocessing finished.")
            except Exception as exc:
                st.error(f"Preprocessing failed: {exc}")

    processed_df = st.session_state.get("processed_df")
    if processed_df is not None:
        show_top_metrics(processed_df)
        st.subheader("Processed data preview")
        st.dataframe(processed_df.head(25), use_container_width=True)
        st.subheader("Preprocessing log")
        st.code("\n".join(st.session_state.get("preprocess_log", [])) or "No extra log available.")
        st.download_button(
            "Download processed_data.csv",
            data=processed_df.to_csv(index=False),
            file_name="processed_data.csv",
            mime="text/csv",
        )
    else:
        st.info("Choose settings and run preprocessing to create your cleaned dataset.")

elif section == "Model Lab":
    st.subheader("Train, compare, and understand models")

    source_choice = st.radio(
        "Training data source",
        ["Raw uploaded data", "Processed data"],
        horizontal=True,
    )
    train_df = raw_df if source_choice == "Raw uploaded data" else current_data(raw_df)
    if source_choice == "Processed data" and st.session_state.get("processed_df") is None:
        st.warning("No processed data found yet, so the raw data is being used.")
        train_df = raw_df

    st.dataframe(train_df.head(20), use_container_width=True)

    target_col = st.selectbox("Target column", train_df.columns)
    default_problem_type = get_default_problem_type(train_df[target_col])
    problem_type = st.radio(
        "Problem type",
        ["classification", "regression"],
        index=0 if default_problem_type == "classification" else 1,
        horizontal=True,
    )

    train_tab, compare_tab = st.tabs(["Train one model", "Compare all models"])

    with train_tab:
        model_choices = get_model_options(problem_type)
        model_name = st.selectbox("Model", list(model_choices.keys()))
        model_key = model_choices[model_name]

        st.markdown("#### Parameters")
        params = {}
        if model_key == "logistic_regression":
            c1, c2 = st.columns(2)
            params["C"] = c1.slider("C", 0.01, 10.0, 1.0)
            params["max_iter"] = c2.slider("max_iter", 100, 2000, 500, step=100)
        elif model_key in {"random_forest_classifier", "random_forest_regressor"}:
            c1, c2, c3 = st.columns(3)
            params["n_estimators"] = c1.slider("n_estimators", 50, 500, 200, step=50)
            params["max_depth"] = c2.selectbox("max_depth", [None, 3, 5, 10, 20], index=0)
            params["min_samples_split"] = c3.slider("min_samples_split", 2, 10, 2)
        elif model_key in {"gradient_boosting_classifier", "gradient_boosting_regressor"}:
            c1, c2 = st.columns(2)
            params["n_estimators"] = c1.slider("n_estimators", 50, 400, 150, step=25)
            params["learning_rate"] = c2.slider("learning_rate", 0.01, 0.50, 0.10, step=0.01)
        elif model_key in {"knn_classifier", "knn_regressor"}:
            c1, c2 = st.columns(2)
            params["n_neighbors"] = c1.slider("n_neighbors", 1, 25, 5)
            params["weights"] = c2.selectbox("weights", ["uniform", "distance"])
        elif model_key == "svc":
            c1, c2 = st.columns(2)
            params["C"] = c1.slider("C", 0.01, 10.0, 1.0)
            params["kernel"] = c2.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
        elif model_key == "svr":
            c1, c2 = st.columns(2)
            params["C"] = c1.slider("C", 0.01, 10.0, 1.0, key="svr_c")
            params["kernel"] = c2.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], key="svr_kernel")
        elif model_key == "ridge":
            params["alpha"] = st.slider("alpha", 0.01, 10.0, 1.0)
        else:
            st.info("This model uses default parameters in this template.")

        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Test size", 0.10, 0.40, 0.20, step=0.05)
        random_state = c2.number_input("Random state", min_value=0, value=42, step=1)
        use_cv = c3.checkbox("Enable cross-validation", value=False)
        cv_folds = st.slider("CV folds", 3, 10, 5)

        if st.button("Train model"):
            try:
                result = fit_and_evaluate(
                    df=train_df,
                    target_col=target_col,
                    problem_type=problem_type,
                    model_key=model_key,
                    params=params,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    use_cv=use_cv,
                    cv_folds=int(cv_folds),
                )

                metrics = result["metrics"]
                metric_cols = st.columns(min(4, len(metrics)))
                for idx, (name, value) in enumerate(metrics.items()):
                    metric_cols[idx % len(metric_cols)].metric(name, f"{value:.4f}")

                if result["cv_scores"] is not None:
                    st.write(
                        f"Cross-validation mean: {result['cv_scores'].mean():.4f} | std: {result['cv_scores'].std():.4f}"
                    )

                preds_df = pd.DataFrame({
                    "actual": result["y_test"],
                    "prediction": result["predictions"],
                }).reset_index(drop=True)
                st.subheader("Prediction preview")
                st.dataframe(preds_df.head(30), use_container_width=True)

                if problem_type == "classification":
                    details = result["details"]
                    cm = pd.DataFrame(
                        details["confusion_matrix"],
                        index=[f"actual_{c}" for c in details["classes"]],
                        columns=[f"pred_{c}" for c in details["classes"]],
                    )
                    st.subheader("Confusion matrix")
                    st.dataframe(cm, use_container_width=True)
                else:
                    plot_df = preds_df.copy()
                    plot_df["residual"] = plot_df["actual"] - plot_df["prediction"]
                    fig = px.scatter(plot_df, x="actual", y="prediction", hover_data=["residual"])
                    st.plotly_chart(fig, use_container_width=True)

                importance_df = result["feature_importance"]
                if not importance_df.empty:
                    st.subheader("Feature importance")
                    fig = px.bar(importance_df.head(15), x="importance", y="feature", orientation="h")
                    st.plotly_chart(fig, use_container_width=True)

                export_payload = {
                    "model": model_name,
                    "problem_type": problem_type,
                    "params": params,
                    "metrics": {k: float(v) for k, v in metrics.items()},
                }
                st.download_button(
                    "Download metrics.json",
                    data=json.dumps(export_payload, indent=2),
                    file_name="model_metrics.json",
                    mime="application/json",
                )
                st.download_button(
                    "Download predictions.csv",
                    data=preds_df.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    with compare_tab:
        c1, c2 = st.columns(2)
        compare_test_size = c1.slider("Leaderboard test size", 0.10, 0.40, 0.20, step=0.05, key="cmp_ts")
        compare_random_state = c2.number_input("Leaderboard random state", min_value=0, value=42, step=1, key="cmp_rs")
        if st.button("Run model comparison"):
            try:
                leaderboard = compare_models(
                    df=train_df,
                    target_col=target_col,
                    problem_type=problem_type,
                    test_size=float(compare_test_size),
                    random_state=int(compare_random_state),
                )
                st.dataframe(leaderboard, use_container_width=True)
                fig = px.bar(leaderboard, x="model", y="score", title="Model leaderboard")
                st.plotly_chart(fig, use_container_width=True)
                st.download_button(
                    "Download leaderboard.csv",
                    data=leaderboard.to_csv(index=False),
                    file_name="model_leaderboard.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")
