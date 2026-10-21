# AI-Assisted Blood Test Analysis Dashboard for Diabetes-Risk Screening

This repository contains the full pipeline for an AI-assisted blood test dashboard: NHANES data preprocessing, an FT-Transformer classifier paired with a reconstruction autoencoder, a deterministic critical-value safety check, an LLM-generated plain-English report, a doctor-in-the-loop approval workflow, and a Streamlit dashboard.

## 1. Requirements

- Python 3.10+ (developed against 3.13)
- A Google Gemini API key (for the LLM narrative step)

Install dependencies from the repo root:

```
pip install -r requirements.txt
```

The two experiment-only scripts under `experiments/` (`cross_architecture_comparison.py`, `hierarchical_experiment.py`) additionally need:

```
pip install xgboost pytorch-tabnet
```

These are not required to run the dashboard or the main pipeline — only to re-run those specific comparison experiments.

## 2. Configure your API key

Copy the example environment file and fill in your own key:

```
cp .env.example .env
```

Edit `.env` and set:

```
GEMINI_API_KEY=your-actual-key-here
```

This is picked up automatically by `python-dotenv` wherever the pipeline runs (`src/main.py`, `dashboard.py`).

## 3. Get the raw data

Place the NHANES `.xpt` files for each two-year cycle under `data/raw/<cycle>/`, one subfolder per cycle:

```
data/raw/1-2/    data/raw/9-10/
data/raw/3-4/    data/raw/11-12/
data/raw/5-6/    data/raw/13-14/
data/raw/7-8/    data/raw/15-16/
                 data/raw/17-20/
                 data/raw/21-23/
```

Each cycle folder needs exactly these five files: `DEMO.xpt`, `BIOPRO.xpt`, `GLU.xpt`, `HDL.xpt`, `GHB.xpt`. These are publicly downloadable from the CDC NHANES website for each survey cycle.

If `data/processed/labeled_dataset.csv` already exists (e.g. you were given a copy of this repo with the processed data included), you can skip straight to Section 5.

## 4. Build the dataset (run once, in this exact order)

From the repo root:

```
python -m src.data_preprocessing.data_pipeline
python -m src.data_preprocessing.holdout
python -m src.data_preprocessing.split_data
python -m src.data_preprocessing.synthetic
python -m src.data_preprocessing.filter_smote
```

What each step does:

- `data_pipeline.py` — merges all ten NHANES cycles, converts units to UK/NHS clinical ranges, derives the diabetes label, and writes `data/processed/labeled_dataset.csv`.
- `holdout.py` — withholds 4 real patients per class (12 total) for dashboard demo use, permanently removing them from the training population.
- `split_data.py` — stratified 70/15/15 train/val/test split, writing `data/processed/{train,val,test}.csv`.
- `synthetic.py` — applies SMOTE to the training split only, writing `data/processed/train_smote.csv` (used by `experiments/cross_architecture_comparison.py`; the production model in Section 5 runs its own separate inline SMOTE and does not depend on this file).
- `filter_smote.py` — filters synthetic Diabetic rows in `train_smote.csv` against the same NICE diagnostic threshold used for real patients.

## 5. Train the production models (run once)

```
python -m src.model.final_models
python -m src.model.train_reconstruction
```

- `final_models.py` trains the binary (Normal vs. At-Risk) FT-Transformer classifier and saves `outputs/models/final_binary_ft.pt` and `outputs/models/final_binary_scaler.pkl`.
- `train_reconstruction.py` trains the reconstruction autoencoder on Normal patients only and saves `outputs/models/reconstruction_autoencoder.pt` and `outputs/models/reconstruction_scaler.pkl`.

These four output files are what the dashboard and inference pipeline load at runtime — retraining is only necessary if you want to reproduce or change the models, not every time you launch the dashboard.

## 6. Run the dashboard

```
streamlit run dashboard.py
```

Default doctor login: username `doctor`, password `changeme123` (hardcoded for this prototype — see `DOCTOR_CREDENTIALS` in `dashboard.py`). Patients log in with a Patient ID only.

A sample upload file is provided at `data/sample_upload/demo_patients.csv` for testing the doctor's batch-upload flow.

## 7. Run the full pipeline once from the command line (no dashboard)

```
python -m src.main
```

This runs one demo holdout patient through the entire pipeline (critical check, prediction, SHAP, anomaly score, LLM narrative, doctor report) and simulates a doctor approval, printing both the doctor and patient report to the console.

## 8. Run the test suites

```
pytest testing/unit_tests.py
pytest testing/integration_tests.py
python testing/e2e_synthetic_tests.py
python testing/report_output_evaluation.py
```

Note: `report_output_evaluation.py` checks whatever patient records currently exist in `outputs/dashboard_store.json` (created the first time you use the dashboard), so its result count depends on how many patients you've processed and approved through the dashboard — it is not a fixed-size test suite like the other three.

## 9. Re-run the supporting experiments (optional)

Everything under `experiments/` is standalone analysis referenced in the dissertation, not required for the dashboard to function. Each can be run directly, e.g.:

```
python -m experiments.age_only_check
python -m experiments.logistic_regression_baseline
python -m experiments.binary_reframing
python -m experiments.cross_architecture_comparison   # requires xgboost, pytorch-tabnet
python -m experiments.rebalancing_comparison
python -m experiments.hierarchical_experiment          # requires xgboost
python -m experiments.clustering_experiment
python -m experiments.correlation_analysis
python -m experiments.compute_anomaly_thresholds
python -m experiments.final_binary_ft_tuning
```

## Project structure

```
src/
  data_preprocessing/   data pipeline, unit conversion, labelling, splitting, SMOTE
  model/                FT-Transformer, reconstruction autoencoder, training scripts
  pipeline/             inference, critical-value check, clinical context, LLM
                         narrative generation, report layout, patient upload
  main.py                run_pipeline() / approve_and_generate_patient_report()
dashboard.py             Streamlit doctor/patient dashboard
experiments/              standalone analysis scripts (not required to run the app)
testing/                   automated test suites
data/                       raw NHANES files, processed CSVs, splits, sample uploads
outputs/                    trained model weights, scalers, logs, dashboard store
```
