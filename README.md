##

Project README


Tools Used:
 * Pandas
 * NumPy
 * SQLite3 (data management)
 * Matplotlib + Seaborn + Plotly (visualizations)
 * Optuna (model tuning)
 * SciKit-Learn
 * LGBM
 * PyTorch
 * Flask/Streamlit


Steps in pipeline:
Scripts build off of one another

(1) python scripts/ingest_data.py
* Ingests team-level data from the NBA API. 

(2) python scripts/clean_data.py
* Cleans raw ingested data

(3) python scripts/make_rolling_features.py
* Makes rolling features that are useful for prediction down the line
* OPTIONAL: Can use python scripts/check_db.py to see what is inside the sqlite DB; also python scripts/rm_table.py for DB mgmt

(4) python scripts/tune_model.py

* OPTIONAL: Can use 'python scripts/check_studies.py' and 'python scripts/visualize_study.py' to check tuning studies in optuna_studies.db;
    also python scripts/rm_study.py for tuning management

(5) python scripts/train_model.py 
* Extracts best hyperparams from tuning ATM, so need to tune a model first

(6) Finally, can run python flask_app.py to run locally run web app and visualize performances of tuned/trained models
* Can navigate all model predictions made for (already played) games made on any date

NOTES:
 * Tuning and training scripts require specifying a modeling_config file (located in modeling_configs; by default 'lasso.py').
 * Fully modular tuning/training setup designed to support many different ML/DL packages + model setups
 * Existing modeling configs are for sklearn models, LGBM model, and DL FFNN model using PyTorch
 * Environment management handled with conda; environment.yml file is included (conda env should be set up before running scripts/project)
 * This repo is a public demo version; has been extended privately