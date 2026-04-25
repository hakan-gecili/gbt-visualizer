# Gradient Boosted Trees Visualizer

An interactive web application for exploring and understanding Gradient Boosted Decision Tree models (LightGBM) at both the tree and ensemble level.

Live Demo: https://gbt-visualizer-frontend.onrender.com/
---

## 🎥 Demo

![Demo](./assets/demo.gif)
-->

---

## Overview

Tree-based ensemble models like LightGBM are powerful but often difficult to interpret.  
This application provides an interactive interface to visualize how predictions are made, both at the individual tree level and across the entire ensemble.

The goal is to make model behavior transparent, intuitive, and explorable in real time.

---

## Key Features

### 🌳 Radial Tree Visualization
- Each tree is displayed in a radial layout
- The exact decision path for a selected observation is highlighted
- Helps understand how the model traverses each tree

### 📊 Per-Tree Contribution Chart
- Shows how each tree contributes to the final prediction (margin space)
- Positive contributions increase the score
- Negative contributions decrease the score

### 🎛️ Interactive Feature Controls
- Modify feature values in real time
- Instantly observe how predictions and decision paths change

### 📁 Dataset Interaction
- Load and inspect datasets
- Select individual observations for analysis

### 📦 Example Loader
- Preconfigured datasets and models (e.g., breast cancer, diabetes)
- Load examples instantly without manual file upload

---

## How It Works

1. A LightGBM model is loaded from a `.txt` file  
2. A dataset is loaded from a `.csv` file  
3. A feature schema file is loaded from a `.json` file
4. For a selected observation:
   - Each tree is evaluated
   - The decision path (root → leaf) is computed
   - Each tree’s contribution to the ensemble prediction is calculated  
5. The application visualizes:
   - Tree structure
   - Active decision paths
   - Per-tree contribution to the final prediction

---

## Tech Stack

- **Frontend:** React + TypeScript (Vite)
- **Backend:** FastAPI (Python)
- **Model:** LightGBM

---

## Running Locally

### Backend

```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload --reload-dir backend/app --reload-exclude ".venv/*" --reload-exclude "venv/*" --reload-exclude "__pycache__/*" --reload-exclude "*.pyc"
```

### Frontend
```bash
npm install
npm run dev
```

## Build Example from CSV

The recommended workflow for new datasets is to generate a complete example folder directly from a training CSV. The helper script trains a LightGBM binary classifier, creates a compact preview dataset, and writes a backend-compatible feature schema in one step.

```bash
python tools/build_example_from_csv.py \
  --data data/train.csv \
  --target-column target \
  --output-dir examples/my_dataset \
  --n-estimators 15 \
  --max-depth 4 \
  --learning-rate 0.1
```

This script generates:

* `model.txt`
* `dataset.csv`
* `feature_schema.json`

Key optional arguments:

* `--max-rows`: limit the exported `dataset.csv` row count for a compact example
* `--inject-missing`: add a few missing values to the exported dataset for UI testing
* `--schema-overrides`: apply a small JSON override file during schema inference
* `--drop-columns`: exclude comma-separated columns from training and export

The generated folder can usually be loaded by the app directly without manual edits.

## Feature Schema Notes

The app supports numeric, binary, categorical, and missing values. `feature_schema.json` controls typed feature inputs and keeps prediction, contribution, and tree-path behavior aligned.

In most cases you should use `tools/build_example_from_csv.py`, which already produces `feature_schema.json` for you. If you only need to regenerate the schema from an existing dataset, `tools/generate_feature_schema.py` is still available as a lower-level helper.

Schema inference remains intentionally conservative:

* schema is inferred from the dataset, not the model file
* ambiguous columns may still require manual overrides
* dataset column names must match model feature names unless overridden

## Deploying On Render

This repository is configured for Render with [render.yaml](./render.yaml).

It deploys as two services:

1. `gbt-visualizer-backend` as a Python web service
2. `gbt-visualizer-frontend` as a static site

### Backend service

- Build command:

```bash
pip install -r backend/requirements.txt
```

- Start command:

```bash
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $PORT
```

### Frontend service

- Build command:

```bash
npm install && npm run build
```

- Publish directory:

```bash
frontend/dist
```

### Usage

* Upload a LightGBM model (.txt)
* Upload a dataset (.csv)
* Optionally upload a feature schema (.json)
* Select an observation
* Explore:
    * Decision paths in trees
    * Tree contributions
    * Impact of feature changes

Or use the Examples dropdown to quickly load preconfigured models and datasets.

## Limitations

* Supports LightGBM models only
* Designed for binary classification
* Supports numeric, binary, categorical, and missing values
* Large models (many trees or deep trees) may reduce visual clarity
* Does not yet include feature importance or SHAP-based explanations

## Roadmap

* Feature importance (global and local)
* SHAP value integration
* Support for XGBoost and Random Forest
* Improved scalability for larger models
* Counterfactual explanation support

## Motivation

Gradient boosted trees are widely used in real-world applications, but their internal decision-making process is often difficult to interpret.

This project aims to provide a clear and interactive way to:

* Understand model behavior
* Debug predictions
* Build trust in machine learning systems

## Author

Hakan Gecili
Senior Data Scientist

GitHub: https://github.com/hakan-gecili

## License

MIT License (or update as needed)
