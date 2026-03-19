# Preplify AI BI Studio

Preplify AI BI Studio is a Streamlit application for end-to-end CSV analytics:
1. Upload and inspect data.
2. Generate quality reports.
3. Build visual analytics (2D and 3D).
4. Run preprocessing pipelines.
5. Train and compare ML models.
6. Detect anomalies.
7. Chat with data and generate BI-style dashboards using Groq LLM.

## Key Features

- Data profile with health score, missing values, and column summary.
- Smart report generation with downloadable Markdown and HTML.
- AI BI dashboard generation from natural-language prompts.
- Visualization lab with histogram, box, scatter, heatmap, PCA 3D, and cluster 3D.
- Preprocessing studio with automatic and custom pipeline modes.
- Model lab for classification and regression with hyperparameter controls.
- Anomaly detection workflow with downloadable outputs.
- Export center for report payload, BI spec, processed data, predictions, anomalies, and leaderboard.

## Tech Stack

- Python 3.10+
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Preplify
- Groq SDK
- python-dotenv

## Project Structure

```text
preplify_streamlit_app/
|-- app.py
|-- requirements.txt
|-- README.md
`-- utils/
    |-- __init__.py
    |-- bi_report.py
    |-- llm.py
    |-- modeling.py
    |-- preplify_bridge.py
    |-- preprocessing.py
    |-- reporting.py
    `-- visuals.py
```

## Installation

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd preplify_streamlit_app
```

### 2) Create virtual environment

Windows (PowerShell):

```powershell
python -m venv zee
.\zee\Scripts\Activate.ps1
```

Windows (cmd):

```cmd
python -m venv zee
zee\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv zee
source zee/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root.

### Required

```env
GROQ_API_KEY=your_groq_api_key
```

### Optional

```env
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com
LLM_MODEL=llama-3.3-70b-versatile
```

Notes:
1. The app is Groq-only (OpenAI code paths removed).
2. Sidebar settings can override env values during runtime.
3. If no API key is present, AI tabs remain available but generation will fail with a clear message.

## Run the App

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## How It Works (Tab by Tab)

### 1) Overview

- Upload a CSV from the sidebar.
- View row/column metrics, health score, sample records, and summary details.

### 2) Report Studio

- Smart Report: Generates report payload from dataset + Preplify recommendations, and supports Markdown and HTML download.
- AI BI Report: Accepts natural-language business prompts, calls Groq LLM to produce dashboard JSON spec, and renders an executive-style BI dashboard.

### 3) Visualization Lab

- Supported charts: Histogram, Box Plot, 2D Scatter, 3D Scatter, 3D PCA, 3D Clusters, Correlation Heatmap, Missing Values chart.

### 4) Preprocessing Studio

- Preplify auto mode: quick baseline cleanup.
- Custom pipeline mode includes missing strategy (mean, median, mode, drop, constant), encoding (onehot, label), scaling (standard, minmax, robust), outlier handling (iqr, zscore), and feature engineering toggle.

### 5) Model Lab

- Select target and task type (classification/regression).
- Choose model and tune parameters from sidebar controls.
- Train selected model and review metrics.
- Compare baseline models and export leaderboard.

### 6) AI Analyst

- Chat with dataset context using Groq.
- Prompt detection for BI request keywords can auto-generate dashboard specs.
- Generated BI dashboard is available in a dedicated subtab.

### 7) Anomaly Lab

- Runs anomaly detection with contamination control.
- Displays summary and row-level anomaly outputs.

### 8) Export Center

- Download outputs: report JSON, AI BI spec JSON, processed dataset CSV, predictions CSV, anomaly results CSV, and model leaderboard CSV.

## LLM Configuration Details

- Provider: Groq only
- Default model: `llama-3.3-70b-versatile`
- Base URL default: `https://api.groq.com`
- Runtime configuration flow:
    1. Sidebar input (if provided)
    2. `.env` values
    3. Internal defaults

## Common Errors and Fixes

### 1) Missing GROQ API key

Error examples:
- "Missing GROQ_API_KEY"
- "Groq selected but no API key found"

Fix:
1. Add `GROQ_API_KEY` in `.env`, or
2. Enter it in sidebar under LLM Settings.

### 2) Duplicate Streamlit plotly element ID

Cause:
- Identical charts rendered multiple times without unique element keys.

Status:
- Fixed in BI dashboard renderer by assigning explicit chart keys.

### 3) CSV fails to load

Fix:
1. Ensure file is valid CSV and UTF-8 compatible.
2. Try a smaller sample file to isolate malformed rows.

## Development Notes

- Main app entry: `app.py`
- LLM integration and prompt handling: `utils/llm.py`
- AI BI dashboard rendering: `utils/bi_report.py`
- Modeling utilities: `utils/modeling.py`
- Visual helpers: `utils/visuals.py`
- Reporting utilities: `utils/reporting.py`
- Preplify bridge and preprocessing: `utils/preplify_bridge.py`, `utils/preprocessing.py`

## Quick Start Checklist

1. Create and activate venv.
2. Install requirements.
3. Add `GROQ_API_KEY` in `.env`.
4. Run `streamlit run app.py`.
5. Upload CSV.
6. Generate report, visuals, and AI BI dashboard.

## License

Add your preferred license text here.
