from __future__ import annotations

import json
import os
from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from utils.bi_report import render_bi_dashboard
from utils.llm import (
    ask_dataset_question,
    generate_bi_report_spec,
    llm_status,
    resolve_llm_config,
)
from utils.modeling import compare_models, detect_anomalies, train_model
from utils.preplify_bridge import (
    apply_auto_prep,
    apply_custom_pipeline,
    get_recommendations,
    get_report,
    status_message,
)
from utils.reporting import (
    basic_profile,
    build_report_payload,
    column_summary,
    health_score,
    infer_task_type,
    report_html,
    report_markdown,
)
from utils.visuals import (
    box_chart,
    cluster_3d,
    correlation_heatmap,
    histogram_chart,
    missing_bar,
    numeric_pair_candidates,
    pca_3d,
    scatter_2d,
    scatter_3d,
)

load_dotenv()

st.set_page_config(
    page_title="DataCanvas Nexus",
    page_icon="✨",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(StringIO(file_bytes.decode("utf-8", errors="ignore")))


def set_state_defaults() -> None:
    defaults = {
        "processed_df": None,
        "report_payload": None,
        "predictions_df": None,
        "anomaly_df": None,
        "leaderboard_df": None,
        "ai_report_spec": None,
        "ai_report_raw": None,
        "chat_history": [],
        "ai_prompt_draft": "",
        "clear_ai_prompt": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def metric_row(metrics: dict) -> None:
    if not metrics:
        st.info("No metrics available yet.")
        return

    cols = st.columns(len(metrics))
    for col, (key, value) in zip(cols, metrics.items()):
        col.metric(key.replace("_", " ").title(), value)


def parameter_controls(task: str, model_name: str) -> dict:
    params = {}

    if task == "classification":
        if model_name == "Logistic Regression":
            params["C"] = st.slider("C", 0.01, 10.0, 1.0, 0.01)
            params["max_iter"] = st.slider("max_iter", 100, 2000, 500, 50)

        elif model_name == "Random Forest":
            params["n_estimators"] = st.slider("n_estimators", 50, 600, 200, 10)
            params["max_depth"] = st.slider("max_depth (0 = None)", 0, 40, 0, 1)
            params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2, 1)

        elif model_name == "Gradient Boosting":
            params["n_estimators"] = st.slider("n_estimators", 50, 600, 150, 10)
            params["learning_rate"] = st.slider("learning_rate", 0.01, 1.0, 0.1, 0.01)

        elif model_name == "KNN":
            params["n_neighbors"] = st.slider("n_neighbors", 1, 30, 5, 1)

        elif model_name == "SVM":
            params["C"] = st.slider("C", 0.01, 10.0, 1.0, 0.01)
            params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])

    else:
        if model_name == "Linear Regression":
            st.caption("Linear Regression has no tuning parameters in this app.")

        elif model_name == "Random Forest":
            params["n_estimators"] = st.slider("n_estimators", 50, 600, 200, 10)
            params["max_depth"] = st.slider("max_depth (0 = None)", 0, 40, 0, 1)
            params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2, 1)

        elif model_name == "Gradient Boosting":
            params["n_estimators"] = st.slider("n_estimators", 50, 600, 150, 10)
            params["learning_rate"] = st.slider("learning_rate", 0.01, 1.0, 0.1, 0.01)

        elif model_name == "KNN":
            params["n_neighbors"] = st.slider("n_neighbors", 1, 30, 5, 1)

        elif model_name == "SVR":
            params["C"] = st.slider("C", 0.01, 10.0, 1.0, 0.01)
            params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])

    return params


def pick_source_df(raw_df: pd.DataFrame, use_processed: bool) -> pd.DataFrame:
    if use_processed and st.session_state.processed_df is not None:
        return st.session_state.processed_df
    return raw_df


def maybe_generate_bi_from_prompt(df: pd.DataFrame, prompt: str, llm_cfg: dict) -> bool:
    lowered = prompt.lower()
    keywords = ["report", "dashboard", "powerbi", "power bi", "tableau", "executive summary"]

    if any(keyword in lowered for keyword in keywords):
        spec, raw = generate_bi_report_spec(
            df=df,
            prompt=prompt,
            provider=llm_cfg["provider"],
            model_name=llm_cfg["model"],
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg["base_url"],
        )
        st.session_state.ai_report_spec = spec
        st.session_state.ai_report_raw = raw
        return True

    return False


def answer_ai_prompt(df: pd.DataFrame, prompt: str, llm_cfg: dict) -> str:
    if not llm_cfg["api_key"]:
        return "Add your API key in the sidebar or select Auto (from env)."

    try:
        generated = maybe_generate_bi_from_prompt(df, prompt, llm_cfg)
        if generated:
            return (
                "I created a BI-style dashboard report below. "
                "Open the 'Generated BI Dashboard' subtab to view it."
            )

        return ask_dataset_question(
            df=df,
            question=prompt,
            provider=llm_cfg["provider"],
            model_name=llm_cfg["model"],
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg["base_url"],
        )

    except Exception as exc:
        return f"AI request failed: {exc}"


def build_llm_sidebar() -> tuple:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">DataCanvas Nexus</div>
                <div class="sidebar-brand-sub">AI analytics cockpit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.header("Workspace")
        st.info(status_message())

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        st.markdown("---")

        st.subheader("AI Engine Settings")

        provider_choice = st.selectbox(
            "LLM Provider",
            ["Auto (from env)", "Groq"],
            index=0,
        )

        default_groq_key = os.getenv("GROQ_API_KEY", "")
        default_groq_base = os.getenv("GROQ_BASE_URL", "https://api.groq.com")
        default_model = (
            os.getenv("LLM_MODEL", "")
            or os.getenv("GROQ_MODEL", "")
            or "llama-3.3-70b-versatile"
        )

        selected_api_key = ""
        selected_base_url = ""
        selected_model = default_model

        if provider_choice == "Auto (from env)":
            st.caption("Uses your already configured environment variables automatically.")
            selected_model = st.text_input("Model override (optional)", value=default_model)

        elif provider_choice == "Groq":
            selected_api_key = st.text_input("GROQ_API_KEY", value=default_groq_key, type="password")
            selected_base_url = st.text_input("Groq Base URL", value=default_groq_base)
            selected_model = st.text_input(
                "Groq model",
                value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            )

        llm_cfg = resolve_llm_config(
            provider_choice=provider_choice,
            manual_api_key=selected_api_key,
            manual_base_url=selected_base_url,
            manual_model=selected_model,
        )

        st.info(llm_status(llm_cfg))
        st.code(
            {
                "provider": llm_cfg["provider"],
                "base_url": llm_cfg["base_url"],
                "has_api_key": bool(llm_cfg["api_key"]),
                "api_key_prefix": llm_cfg["api_key"][:5] if llm_cfg["api_key"] else "",
            },
            language="python",
        )

    return uploaded_file, llm_cfg


def main() -> None:
    set_state_defaults()

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700&display=swap');

        :root {
            --brand-deep: #0f172a;
            --brand-mid: #1d3557;
            --brand-accent: #ff6b35;
            --brand-accent-2: #22c55e;
            --brand-accent-3: #06b6d4;
            --card: #ffffff;
            --ink: #0b1320;
            --muted: #4b5563;
            --border: #dfe6f2;
            --sidebar-bg: #ffffff;
        }

        .stApp {
            font-family: 'Manrope', sans-serif;
            color: var(--ink);
            background:
                radial-gradient(1200px 520px at 110% -100px, rgba(34, 197, 94, 0.16), transparent 58%),
                radial-gradient(980px 460px at -5% -140px, rgba(6, 182, 212, 0.2), transparent 60%),
                linear-gradient(180deg, #f8fbff 0%, #eef6f9 100%);
        }

        section[data-testid="stSidebar"] {
            background: var(--sidebar-bg) !important;
            border-right: 1px solid #e5edf7;
        }

        section[data-testid="stSidebar"] * {
            color: #111827 !important;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: #f8fafc !important;
            border: 1px solid #dbe3ef !important;
            border-radius: 12px;
        }

        section[data-testid="stSidebar"] [data-testid="stAlert"] {
            background: #eef2ff !important;
            border: 1px solid #c7d2fe !important;
            border-radius: 12px;
        }

        h1, h2, h3, .stTabs [data-baseweb="tab"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            letter-spacing: -0.01em;
        }

        .sidebar-brand {
            border: 1px solid #dbeafe;
            border-radius: 16px;
            background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
            padding: 12px 14px;
            margin-bottom: 14px;
        }

        .sidebar-brand-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800;
            color: #0f172a;
            font-size: 15px;
        }

        .sidebar-brand-sub {
            color: #334155;
            font-size: 12px;
            margin-top: 2px;
        }

        .hero-box {
            padding: 26px;
            border-radius: 24px;
            background:
                radial-gradient(500px 180px at 95% 0%, rgba(255,255,255,0.2), transparent 70%),
                linear-gradient(128deg, var(--brand-deep), var(--brand-mid));
            color: white;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 18px 44px rgba(15, 23, 42, 0.26);
            animation: hero-enter 550ms ease-out;
        }

        .hero-sub {
            color: #e2ebff;
            margin-top: 8px;
            max-width: 900px;
            line-height: 1.55;
        }

        .hero-strip {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 14px;
        }

        .hero-pill {
            border: 1px solid rgba(255,255,255,0.22);
            padding: 7px 11px;
            border-radius: 999px;
            font-size: 12px;
            color: #fff;
            background: rgba(255,255,255,0.08);
        }

        .hero-kicker {
            display: inline-block;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #dbeafe;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(219, 234, 254, 0.25);
            background: rgba(219, 234, 254, 0.1);
        }

        .block-container {
            padding-top: 1.2rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #eaf4f8;
            border-radius: 14px;
            padding: 6px;
            border: 1px solid #d4e6ef;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 8px 14px;
            color: #35425d;
            font-weight: 600;
            transition: all 120ms ease;
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff;
            color: #0f172a;
            box-shadow: 0 2px 14px rgba(2, 6, 23, 0.1);
        }

        .stTabs [data-baseweb="tab-panel"] {
            color: var(--ink);
        }

        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 13px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }

        div[data-testid="stMetricLabel"] p {
            color: #5b6475 !important;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
        }

        div[data-testid="stMetricDelta"] {
            color: #1e293b !important;
        }

        div[data-testid="stForm"],
        div[data-testid="stExpander"] {
            border-radius: 14px;
            border: 1px solid var(--border);
            background: #ffffff;
        }

        /* Ensure widget labels and option text stay readable on light surfaces */
        label[data-testid="stWidgetLabel"] p,
        div[role="radiogroup"] label,
        div[role="radiogroup"] label p,
        div[role="radiogroup"] label span,
        div[data-baseweb="checkbox"] label,
        div[data-baseweb="checkbox"] span,
        div[data-baseweb="select"] *,
        div[data-baseweb="input"] input,
        textarea,
        .stMarkdown,
        .stCaption,
        .stText {
            color: var(--ink) !important;
        }

        .stAlert p,
        .stInfo p,
        .stWarning p,
        .stSuccess p,
        .stError p {
            color: #111827 !important;
        }

        /* Input surface + text contrast fix (main area + sidebar + chat input) */
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        textarea,
        input {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #cfd8e8 !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input,
        textarea,
        input {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }

        div[data-testid="stTextArea"] textarea {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #b8cbe6 !important;
            border-radius: 10px !important;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 1px #3b82f6 !important;
            outline: none !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #6b7280 !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] input {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #94a3b8 !important;
        }

        div[data-testid="stForm"] {
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stExpander"] details summary,
        div[data-testid="stExpander"] details summary p {
            color: #0f172a !important;
            font-weight: 600;
        }

        .stRadio > div {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 12px;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid #cfddee;
            font-weight: 600;
            transition: transform 130ms ease, box-shadow 130ms ease, border-color 130ms ease;
            background: #f8fbff;
            color: #0f172a !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: #9fb6d1;
            box-shadow: 0 9px 20px rgba(15, 23, 42, 0.11);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
            color: #ffffff !important;
            border: none;
        }

        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.34);
        }

        .stButton > button[kind="secondary"] {
            background: #e9f1fb !important;
            color: #0f172a !important;
            border: 1px solid #bfd2eb !important;
        }

        @keyframes hero-enter {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 900px) {
            .hero-box {
                padding: 17px;
                border-radius: 16px;
            }
            .hero-sub {
                font-size: 14px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 7px 10px;
                font-size: 12px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-box">
            <span class="hero-kicker">Next-gen analytics workspace</span>
            <h1 style="margin:10px 0 6px 0;">✨ DataCanvas Nexus</h1>
            <div class="hero-sub">
                A bold data studio for teams that want clean prep, visual intelligence, and AI-powered analysis in one experience.
            </div>
            <div class="hero-strip">
                <span class="hero-pill">From CSV to decisions</span>
                <span class="hero-pill">AI Analyst that understands your dataset</span>
                <span class="hero-pill">Built-in ML + anomaly intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file, llm_cfg = build_llm_sidebar()

    if uploaded_file is None:
        st.warning("Upload a CSV file to start.")
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
        return

    df = load_csv(uploaded_file.getvalue())
    profile = basic_profile(df)
    health = health_score(df)

    overview_tab, report_tab, visual_tab, prep_tab, model_tab, ai_tab, anomaly_tab, export_tab = st.tabs(
        [
            "Pulse Overview",
            "Insight Reports",
            "Visual Atlas",
            "Data Refinery",
            "Model Forge",
            "AI Analyst",
            "Outlier Radar",
            "Export Vault",
        ]
    )

    with overview_tab:
        st.caption("Your dataset pulse at a glance: shape, health, and quick structural signals.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", profile["rows"])
        c2.metric("Columns", profile["columns"])
        c3.metric("Missing Cells", profile["missing_total"])
        c4.metric("Health Score", f"{health['score']}/100")

        left, right = st.columns([1.4, 1])

        with left:
            st.subheader("Data Preview")
            st.dataframe(df.head(20), use_container_width=True)

        with right:
            st.subheader("Health Details")
            st.json(health)
            st.subheader("Column Summary")
            st.dataframe(column_summary(df), use_container_width=True, hide_index=True)

    with report_tab:
        st.caption("Create boardroom-ready summaries or AI-crafted dashboard specs in one workspace.")
        smart_subtab, ai_bi_subtab = st.tabs(["Smart Report", "AI BI Report"])

        with smart_subtab:
            st.subheader("Generate a standard report")

            if st.button("Build smart report", type="primary"):
                preplify_report = get_report(df)
                recommendations = get_recommendations(df)
                st.session_state.report_payload = build_report_payload(df, preplify_report, recommendations)

            if st.session_state.report_payload is not None:
                payload = st.session_state.report_payload

                metric_row(
                    {
                        "Health Score": payload["health"]["score"],
                        "Missing Ratio": payload["health"]["missing_ratio"],
                        "Duplicate Ratio": payload["health"]["duplicate_ratio"],
                        "Outlier Ratio": payload["health"]["outlier_ratio"],
                    }
                )

                st.markdown("### Recommendations")
                for item in payload.get("recommendations", []):
                    st.write(f"- {item}")

                st.markdown("### Preplify Report Output")
                st.json(payload.get("preplify_report", {}))

                md = report_markdown(payload)
                html = report_html(payload)

                st.download_button(
                    "Download Markdown report",
                    md,
                    file_name="smart_dataset_report.md",
                )
                st.download_button(
                    "Download HTML report",
                    html,
                    file_name="smart_dataset_report.html",
                )
            else:
                st.info("Click the button to build the report.")

        with ai_bi_subtab:
            st.subheader("Generate a Tableau / Power BI style report with AI")
            st.caption(
                "Example request: Create an executive sales dashboard with KPI cards, category trends, and a 3D exploration view."
            )

            ai_request = st.text_area(
                "AI report request",
                value="Create an executive dashboard with KPI cards, key insights, and business-friendly charts.",
                height=120,
            )

            use_processed_for_ai = st.toggle("Use processed dataframe for AI report", value=False)
            report_df = pick_source_df(df, use_processed_for_ai)

            if st.button("Generate AI BI report", type="primary"):
                if not llm_cfg["api_key"]:
                    st.error("Add a valid API key or use Auto (from env).")
                else:
                    try:
                        spec, raw = generate_bi_report_spec(
                            df=report_df,
                            prompt=ai_request,
                            provider=llm_cfg["provider"],
                            model_name=llm_cfg["model"],
                            api_key=llm_cfg["api_key"],
                            base_url=llm_cfg["base_url"],
                        )
                        st.session_state.ai_report_spec = spec
                        st.session_state.ai_report_raw = raw
                    except Exception as exc:
                        st.error(f"AI BI report generation failed: {exc}")

            if st.session_state.ai_report_spec is not None:
                render_bi_dashboard(
                    report_df,
                    st.session_state.ai_report_spec,
                    key_prefix="report_studio_bi",
                )
                with st.expander("Raw AI response / JSON"):
                    st.code(
                        st.session_state.ai_report_raw or json.dumps(st.session_state.ai_report_spec, indent=2),
                        language="json",
                    )
            else:
                st.info("Generate an AI BI report to see a Power BI-like dashboard layout here.")

    with visual_tab:
        st.subheader("2D + 3D visualization workspace")
        st.caption("Explore patterns, segments, and relationships with interactive visual storytelling.")

        use_processed = st.toggle("Use processed dataframe for visualizations", value=False)
        vis_df = pick_source_df(df, use_processed)

        numeric_cols = numeric_pair_candidates(vis_df)
        all_cols = list(vis_df.columns)

        visual_mode = st.radio(
            "Choose visualization",
            [
                "Histogram",
                "Box Plot",
                "2D Scatter",
                "3D Scatter",
                "3D PCA",
                "3D Clusters",
                "Correlation Heatmap",
                "Missing Values",
            ],
            horizontal=True,
        )

        fig = None

        if visual_mode == "Histogram" and numeric_cols:
            col = st.selectbox("Column", numeric_cols, key="hist_col")
            fig = histogram_chart(vis_df, col)

        elif visual_mode == "Box Plot" and numeric_cols:
            y_col = st.selectbox("Numeric column", numeric_cols, key="box_y")
            color_col = st.selectbox("Color by", [None] + all_cols, key="box_color")
            fig = box_chart(vis_df, y_col, color_col)

        elif visual_mode == "2D Scatter" and len(numeric_cols) >= 2:
            x_col = st.selectbox("X axis", numeric_cols, key="s2_x")
            y_col = st.selectbox("Y axis", [c for c in numeric_cols if c != x_col], key="s2_y")
            color_col = st.selectbox("Color by", [None] + all_cols, key="s2_color")
            size_col = st.selectbox("Size by", [None] + numeric_cols, key="s2_size")
            fig = scatter_2d(vis_df, x_col, y_col, color_col, size_col)

        elif visual_mode == "3D Scatter" and len(numeric_cols) >= 3:
            x_col = st.selectbox("X axis", numeric_cols, key="s3_x")
            remaining = [c for c in numeric_cols if c != x_col]
            y_col = st.selectbox("Y axis", remaining, key="s3_y")
            remaining = [c for c in remaining if c != y_col]
            z_col = st.selectbox("Z axis", remaining, key="s3_z")
            color_col = st.selectbox("Color by", [None] + all_cols, key="s3_color")
            size_col = st.selectbox("Size by", [None] + numeric_cols, key="s3_size")
            fig = scatter_3d(vis_df, x_col, y_col, z_col, color_col, size_col)

        elif visual_mode == "3D PCA":
            color_col = st.selectbox("Color by", [None] + all_cols, key="pca_color")
            fig = pca_3d(vis_df, color_col)

        elif visual_mode == "3D Clusters":
            n_clusters = st.slider("Number of clusters", 2, 10, 4)
            fig = cluster_3d(vis_df, n_clusters)

        elif visual_mode == "Correlation Heatmap":
            fig = correlation_heatmap(vis_df)

        elif visual_mode == "Missing Values":
            fig = missing_bar(vis_df)

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("This chart needs more compatible numeric columns or non-empty data.")

    with prep_tab:
        st.subheader("Preprocess your data")
        st.caption("Refine raw data into model-ready inputs using automatic or custom pipelines.")

        prep_mode = st.radio("Mode", ["Preplify auto_prep", "Custom pipeline"], horizontal=True)

        if prep_mode == "Preplify auto_prep":
            if st.button("Run auto preprocessing", type="primary"):
                st.session_state.processed_df = apply_auto_prep(df)
        else:
            missing_strategy = st.selectbox(
                "Missing strategy",
                ["mean", "median", "mode", "drop", "constant"],
                index=1,
            )
            encoding = st.selectbox("Encoding", ["onehot", "label"], index=0)
            scaling = st.selectbox("Scaling", ["standard", "minmax", "robust"], index=0)
            outlier_method = st.selectbox("Outlier method", [None, "iqr", "zscore"], index=0)
            feature_engineering = st.toggle("Feature engineering", value=False)

            if st.button("Run custom preprocessing", type="primary"):
                st.session_state.processed_df = apply_custom_pipeline(
                    df,
                    missing_strategy=missing_strategy,
                    encoding=encoding,
                    scaling=scaling,
                    outlier_method=outlier_method,
                    feature_engineering=feature_engineering,
                )

        processed_df = st.session_state.processed_df

        if processed_df is not None:
            st.success("Preprocessing complete.")

            c1, c2 = st.columns(2)
            c1.metric("Original Shape", str(df.shape))
            c2.metric("Processed Shape", str(processed_df.shape))

            st.dataframe(processed_df.head(20), use_container_width=True)

            st.download_button(
                "Download processed CSV",
                processed_df.to_csv(index=False).encode("utf-8"),
                file_name="processed_data.csv",
                mime="text/csv",
            )
        else:
            st.info("Run preprocessing to create a cleaned dataset.")

    with model_tab:
        st.subheader("Train and compare models")
        st.caption("Choose source data, set target/task, tune parameters, and compare model performance.")

        controls_top_left, controls_top_right = st.columns([1.2, 1])

        with controls_top_left:
            source_choice = st.radio(
                "Training data source",
                ["Original dataframe", "Processed dataframe"],
                horizontal=True,
            )

        train_df = pick_source_df(df, source_choice == "Processed dataframe")

        if train_df is None or train_df.empty:
            st.warning("No data available for training.")
        else:
            with controls_top_left:
                target_col = st.selectbox("Target column", train_df.columns)
                task_guess = infer_task_type(train_df[target_col])
                task_options = [task_guess] + [t for t in ["classification", "regression"] if t != task_guess]
                task = st.selectbox("Task type", task_options)
                test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

            with controls_top_right:
                st.markdown("### Model Selection")
                if task == "classification":
                    model_name_select = st.selectbox(
                        "Model",
                        ["Logistic Regression", "Random Forest", "Gradient Boosting", "KNN", "SVM"],
                        key="model_class",
                    )
                else:
                    model_name_select = st.selectbox(
                        "Model",
                        ["Linear Regression", "Random Forest", "Gradient Boosting", "KNN", "SVR"],
                        key="model_reg",
                    )
                st.info(f"Selected model: {model_name_select}")

            with st.expander("Model Hyperparameters", expanded=True):
                params = parameter_controls(task, model_name_select)

            c_left, c_right = st.columns(2)

            if c_left.button("Train selected model", type="primary"):
                try:
                    result = train_model(train_df, target_col, task, model_name_select, params, test_size)
                    st.session_state.predictions_df = result.predictions

                    st.success("Model trained successfully.")
                    metric_row(result.metrics)

                    if result.feature_importance is not None and not result.feature_importance.empty:
                        top_imp = result.feature_importance.head(20)
                        fig_imp = px.bar(
                            top_imp,
                            x="importance",
                            y="feature",
                            orientation="h",
                            title="Feature Importance",
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                    if result.confusion is not None:
                        st.dataframe(result.confusion, use_container_width=True)

                    st.dataframe(result.predictions.head(25), use_container_width=True)

                except Exception as exc:
                    st.error(f"Training failed: {exc}")

            if c_right.button("Compare baseline models"):
                try:
                    leaderboard = compare_models(train_df, target_col, task, test_size)
                    st.session_state.leaderboard_df = leaderboard
                    st.dataframe(leaderboard, use_container_width=True)

                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")

            if st.session_state.leaderboard_df is not None:
                st.markdown("### Latest leaderboard")
                st.dataframe(st.session_state.leaderboard_df, use_container_width=True)

    with ai_tab:
        chat_subtab, dashboard_subtab = st.tabs(["Chat with Data", "Generated BI Dashboard"])

        with chat_subtab:
            st.subheader("Ask the AI analyst")
            st.caption("Type a question below to start the analysis.")

            use_processed_chat = st.toggle("Use processed dataframe in chat", value=False)
            chat_df = pick_source_df(df, use_processed_chat)

            if st.session_state.clear_ai_prompt:
                st.session_state.ai_prompt_draft = ""
                st.session_state.clear_ai_prompt = False

            for item in st.session_state.chat_history:
                with st.chat_message(item["role"]):
                    st.markdown(item["content"])

            st.text_area(
                "Your question",
                key="ai_prompt_draft",
                height=90,
                placeholder="Type your own question here. Example: What are the top 3 columns hurting data quality?",
            )
            ask_col, clear_col = st.columns([1, 1])
            ask_now = ask_col.button("Ask AI", type="primary", use_container_width=True)
            clear_chat = clear_col.button("Clear chat", use_container_width=True)

            if clear_chat:
                st.session_state.chat_history = []
                st.session_state.clear_ai_prompt = True
                st.rerun()

            if ask_now:
                cleaned_prompt = st.session_state.ai_prompt_draft.strip()

                if not cleaned_prompt:
                    st.warning("Please type your question before clicking Ask AI.")
                else:
                    st.session_state.chat_history.append({"role": "user", "content": cleaned_prompt})

                    with st.chat_message("user"):
                        st.markdown(cleaned_prompt)

                    with st.spinner("Analyzing your dataset..."):
                        answer = answer_ai_prompt(chat_df, cleaned_prompt, llm_cfg)

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.clear_ai_prompt = True
                    st.rerun()

                    with st.chat_message("assistant"):
                        st.markdown(answer)

        with dashboard_subtab:
            st.subheader("Latest AI-generated BI dashboard")
            st.caption("Any dashboard created from AI Report or Ask AI appears here automatically.")

            use_processed_dash = st.toggle(
                "Use processed dataframe in BI dashboard",
                value=False,
                key="dash_processed",
            )
            ai_df = pick_source_df(df, use_processed_dash)

            if st.session_state.ai_report_spec is not None:
                render_bi_dashboard(
                    ai_df,
                    st.session_state.ai_report_spec,
                    key_prefix="ai_tab_bi",
                )
            else:
                st.info("Ask the AI to make a report, dashboard, Power BI report, or Tableau-style report.")

    with anomaly_tab:
        st.subheader("Detect anomalies")

        anomaly_source = st.radio(
            "Anomaly data source",
            ["Original dataframe", "Processed dataframe"],
            horizontal=True,
        )
        anomaly_df = pick_source_df(df, anomaly_source == "Processed dataframe")
        contamination = st.slider("Contamination", 0.01, 0.30, 0.05, 0.01)

        if st.button("Run anomaly detection", type="primary"):
            try:
                anomaly_output, summary = detect_anomalies(anomaly_df, contamination)
                st.session_state.anomaly_df = anomaly_output
                st.dataframe(summary, use_container_width=True)
                st.dataframe(anomaly_output.head(30), use_container_width=True)
            except Exception as exc:
                st.error(f"Anomaly detection failed: {exc}")

    with export_tab:
        st.subheader("Download outputs")

        if st.session_state.report_payload is not None:
            st.download_button(
                "Download report JSON",
                json.dumps(st.session_state.report_payload, indent=2),
                file_name="report_payload.json",
                mime="application/json",
            )

        if st.session_state.ai_report_spec is not None:
            st.download_button(
                "Download AI BI spec JSON",
                json.dumps(st.session_state.ai_report_spec, indent=2),
                file_name="ai_bi_report_spec.json",
                mime="application/json",
            )

        if st.session_state.processed_df is not None:
            st.download_button(
                "Download processed dataset",
                st.session_state.processed_df.to_csv(index=False).encode("utf-8"),
                file_name="processed_dataset.csv",
                mime="text/csv",
            )

        if st.session_state.predictions_df is not None:
            st.download_button(
                "Download predictions",
                st.session_state.predictions_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

        if st.session_state.anomaly_df is not None:
            st.download_button(
                "Download anomaly results",
                st.session_state.anomaly_df.to_csv(index=False).encode("utf-8"),
                file_name="anomaly_results.csv",
                mime="text/csv",
            )

        if st.session_state.leaderboard_df is not None:
            st.download_button(
                "Download model leaderboard",
                st.session_state.leaderboard_df.to_csv(index=False).encode("utf-8"),
                file_name="model_leaderboard.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()