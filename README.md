# Preplify Streamlit ML Studio

A Streamlit web app that lets a user:
- upload a CSV dataset
- create a report
- visualize the data
- preprocess the dataset with Preplify
- choose a machine learning model
- tune model parameters
- train and evaluate the model

## Project structure

```text
preplify_streamlit_app/
├── app.py
├── requirements.txt
├── README.md
└── utils/
    ├── modeling.py
    ├── preprocessing.py
    ├── reporting.py
    └── visuals.py
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- Preplify is used for preprocessing and report/recommendation hooks.
- Scikit-learn is used for model training and evaluation.
- If your data is very large, start with a smaller CSV first.
