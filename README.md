# Gradient Boosted Trees Visualizer

An interactive web application for exploring and understanding Gradient
Boosted Decision Tree models (**LightGBM and XGBoost**) at both the tree
and ensemble level.

Live Demo: https://gbt-visualizer-frontend.onrender.com/

------------------------------------------------------------------------

## 🎥 Demo

![Demo](./assets/demo.gif)

------------------------------------------------------------------------

## Overview

Tree-based ensemble models like LightGBM and XGBoost are powerful but
often difficult to interpret.\
This application provides an interactive interface to visualize how
predictions are made, both at the individual tree level and across the
entire ensemble.

The goal is to make model behavior **transparent, intuitive, and
explorable in real time**.

------------------------------------------------------------------------

## 🆕 Version 4

- Added a 3D cube to visualize feature usage across trees and depths
- Replaced layered outcome views with directional bars for positive/negative signals
- Enabled tree-level filtering so all views reflect the selected tree
- Synchronized tree selection across all panels
- Improved path highlighting without losing underlying feature information
- Simplified visuals for better clarity and interpretability


## Version 3

Version 3 adds counterfactuals and makes the app interactive beyond visualization.

### Counterfactuals
- Generate feature changes that flip predictions  
- Works for both **LightGBM** and **XGBoost**  
- Uses the current feature values (not just dataset rows)

### Threshold Control
- Threshold is user-controlled (0.0–1.0)  
- Predictions and counterfactuals use this value  

### Counterfactual Paths
- Shows the actual path after applying counterfactual changes  
- Differences from current path are visible  
- Also rendered on:
  - Radial tree view  
  - Selected tree panel  

### Performance
- Faster counterfactual evaluation using batch prediction  
- Enabled with:

```
USE_FAST_CF_EVALUATOR=1
```

### LightGBM Improvement
- Replaced fixed search limit with adaptive search steps  
- Finds counterfactuals more reliably  

------------------------------------------------------------------------

## 🚀 Key Features

### 🌳 Radial Tree Visualization

-   Each tree is displayed in a radial layout
-   Active decision path is highlighted for a selected observation
-   Helps understand how each tree contributes to the prediction

------------------------------------------------------------------------

### 3D Visualization (Ensemble Structure Cubes)

- 3D visualization of feature usage across trees and depths
- Shows how features contribute to positive and negative outcomes via directional signals
- Supports per-tree exploration with consistent filtering across views
- Highlights active decision paths for selected observations
- Designed for intuitive exploration with minimal visual clutter

------------------------------------------------------------------------

### 🌲 Selected Tree Panel

-   Visualizes a single tree in a top-down structure
-   Highlights the exact path taken (root → leaf)
-   Displays correct split semantics:
    -   LightGBM: `<=`
    -   XGBoost: `<`
    -   Categorical: `in {…}`
-   Supports zooming and panning

------------------------------------------------------------------------

### 🧭 Decision Path Visualization

-   Clearly shows how a sample traverses each tree
-   Consistent with backend traversal logic
-   Handles:
    -   numeric splits
    -   categorical splits
    -   missing-value routing

------------------------------------------------------------------------

### 📊 Per-Tree Contribution Chart

-   Displays each tree's contribution to the final prediction (margin
    space)
-   Positive → increases prediction
-   Negative → decreases prediction

------------------------------------------------------------------------

### 📈 Feature Importance Panel (NEW)

-   Toggle between:
    -   **Global importance** (model-wide)
    -   **Local importance** (current prediction)
-   Helps understand which features matter most

------------------------------------------------------------------------

### 🎛️ Interactive Feature Controls

-   Modify feature values in real time
-   Instantly observe:
    -   prediction changes
    -   path changes
    -   tree contributions

------------------------------------------------------------------------

### 📁 Dataset Interaction

-   Load datasets
-   Select individual observations
-   Syncs with feature controls and visualizations

------------------------------------------------------------------------

### 📦 Example Loader (Improved)

-   Organized by dataset and model type:

examples/`<dataset>`{=html}/`<model_family>`{=html}/

-   Easily compare:
    -   LightGBM vs XGBoost on the same dataset
-   Compact UI with dataset/model selection

------------------------------------------------------------------------

## 🧠 How It Works

1.  A model is loaded:
    -   LightGBM (`.txt`)
    -   XGBoost (`.json`)
2.  A dataset is loaded (`.csv`)
3.  A feature schema is loaded (`.json`)
4.  For a selected observation:
    -   Each tree is evaluated
    -   Decision path is computed
    -   Tree contributions are calculated
5.  The app visualizes:
    -   Tree structure
    -   Active paths
    -   Contributions
    -   Feature importance

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   **Frontend:** React + TypeScript (Vite)
-   **Backend:** FastAPI (Python)
-   **Models:**
    -   LightGBM
    -   XGBoost

------------------------------------------------------------------------

## 🛠 Running Locally

### Backend

pip install -r backend/requirements.txt\
uvicorn app.main:app --app-dir backend --reload

### Frontend

npm install\
npm run dev

------------------------------------------------------------------------

## 🏗 Build Example from CSV

python tools/build_example_from_csv.py\
--csv data/train.csv\
--target target\
--model-family xgboost\
--output-dir examples/my_dataset/xgboost

Generates:

-   model.txt / model.json\
-   dataset.csv\
-   feature_schema.json\
-   metadata.json

------------------------------------------------------------------------

## 📐 Feature Schema Notes

The app supports: - numeric - binary - categorical - missing values

Schema is inferred from dataset and ensures consistency across: -
prediction - traversal - visualization

------------------------------------------------------------------------

## 📌 Usage

-   Upload model + dataset + schema\
-   OR use built-in examples

Explore: - Tree structure\
- Decision paths\
- Feature effects\
- Model behavior

------------------------------------------------------------------------

## ⚠️ Limitations

-   Focused on tree ensembles (LightGBM + XGBoost)\
-   Primarily binary classification\
-   Large models may reduce visual clarity\
-   Counterfactuals still experimental

------------------------------------------------------------------------

## 🗺 Roadmap

-   Counterfactual explanations (in progress)\
-   Model comparison mode (LGBM vs XGB side-by-side)\
-   Better categorical handling\
-   Support for:
    -   Random Forest\
    -   CatBoost\
-   SHAP integration

------------------------------------------------------------------------

## 💡 Motivation

Gradient boosted trees are widely used in production, but their decision
logic is often opaque.

This project aims to: - make models interpretable\
- help debug predictions\
- build trust in ML systems

------------------------------------------------------------------------

## 👤 Author

Hakan Gecili\
Senior Data Scientist

GitHub: https://github.com/hakan-gecili

------------------------------------------------------------------------

## 📄 License

MIT License
