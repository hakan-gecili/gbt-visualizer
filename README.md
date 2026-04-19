# Gradient Boosted Trees Visualizer

An interactive web application for exploring and understanding Gradient Boosted Decision Tree models (LightGBM) at both the tree and ensemble level.

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
3. For a selected observation:
   - Each tree is evaluated
   - The decision path (root → leaf) is computed
   - Each tree’s contribution to the ensemble prediction is calculated  
4. The application visualizes:
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
uvicorn app.main:app --app-dir backend --reload
```

### Frontend
```bash
npm install
npm run dev
```

The frontend reads `VITE_API_BASE_URL` from the environment. For local development you can leave it unset if you proxy API requests through the same origin, or point it at your backend explicitly:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

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

### Required Render environment variable

Set this on the frontend Render service after the backend URL is known:

```bash
VITE_API_BASE_URL=https://<your-backend-service>.onrender.com
```

If you create services from the blueprint, Render will prompt you to fill this value because it is marked with `sync: false`.

### Usage

* Upload a LightGBM model (.txt)
* Upload a dataset (.csv)
* Select an observation
* Explore:
    * Decision paths in trees
    * Tree contributions
    * Impact of feature changes

Or use the Examples dropdown to quickly load preconfigured models and datasets.

## Limitations

* Supports LightGBM models only
* Designed for binary classification
* Assumes numerical input features
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
