# NBA Data Modeling Demo

A modular, end-to-end NBA analytics pipeline for data ingestion,
cleaning, feature engineering, model tuning, training, and interactive
visualization via Flask/Streamlit.

------------------------------------------------------------------------

## Tools & Libraries Used

-   **Pandas**, **NumPy**
-   **SQLite3** (data storage + management)
-   **Matplotlib**, **Seaborn**, **Plotly** (visualizations)
-   **Optuna** (hyperparameter tuning)
-   **Scikit-Learn**
-   **LightGBM (LGBM)** 
-   **PyTorch** (FFNN deep learning model)
-   **Flask / Streamlit** (web applications)

------------------------------------------------------------------------

## Project Structure

    NBA-Data-Modeling-Demo/
    │
    ├── data/
    │   ├── raw/
    │   │   └── raw.parquet
    │   └── clean/
    │       └── cleaned_data.db
    │
    ├── modeling_configs/
    │   ├── baseline_0.py
    │   ├── baseline_1.py
    │   ├── lasso.py
    │   ├── lgbm.py
    │   └── torch_ffnn.py
    │
    ├── scripts/
    │   ├── ingest_data.py
    │   ├── clean_data.py
    │   ├── make_rolling_features.py
    │   ├── tune_model.py
    │   ├── train_model.py
    │   ├── check_db.py
    │   ├── check_studies.py
    │   ├── visualize_study.py
    │   ├── rm_table.py
    │   └── rm_study.py
    │
    ├── src/
    │   ├── data/
    │   ├── model/
    │   └── utils.py
    │
    ├── models/
    │── optuna_studies.db
    ├── flask_app.py
    ├── streamlit_app.py
    ├── config.yaml
    └── environment.yml

------------------------------------------------------------------------

## Pipeline Overview

All scripts are modular and build on each other.

### **1. Ingest Raw Data**

    python scripts/ingest_data.py

Fetches team-level NBA data from the NBA API and stores it in
`data/raw`.

------------------------------------------------------------------------

### **2. Clean Data**

    python scripts/clean_data.py

Cleans the raw parquet and loads it into **cleaned_data.db** (SQLite).

------------------------------------------------------------------------

### **3. Generate Rolling Features**

    python scripts/make_rolling_features.py

Creates rolling statistical features for predictive modeling.

Optional database utilities:

    python scripts/check_db.py
    python scripts/rm_table.py

------------------------------------------------------------------------

### **4. Tune Model with Optuna**

    python scripts/tune_model.py

Optional tuning management:

    python scripts/check_studies.py
    python scripts/visualize_study.py
    python scripts/rm_study.py

------------------------------------------------------------------------

### **5. Train Model**

    python scripts/train_model.py

Uses best hyperparameters found through tuning.

Requires selecting a modeling config from `modeling_configs/`.

------------------------------------------------------------------------

### **6. Launch Local Web App**

    python flask_app.py

View: - Model prediction performance\
- Predictions across historical game dates\
- Model-by-model comparison

Streamlit alternative:

    python streamlit_app.py

------------------------------------------------------------------------

## Modeling Configurations

Model setups are modular and can be swapped by editing `config.yaml`.

Included configs: - Baseline models (`baseline_0.py`, `baseline_1.py`)
- Lasso regression (`lasso.py`) - LightGBM (`lgbm.py`)
- PyTorch feedforward network (`torch_ffnn.py`)

The framework is designed to support: - Any sklearn model\
- Advanced tree models\
- Custom deep learning architectures

------------------------------------------------------------------------

## Environment Setup

    conda env create -f environment.yml
    conda activate nba

------------------------------------------------------------------------

## Notes

-   Tuning must occur before training so the trainer can load optimized
    hyperparameters.
-   This repository is a **public demo version** of a private repo with
    more substantial data ingestion, feature engineering, and modeling
    techniques achieving more competitive accuracy



------------------------------------------------------------------------

## License

MIT License (or update as appropriate)
