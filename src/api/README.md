---
title: Football Api
emoji: 📊
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
short_description: FastAPI endpoints
---
# API
This folder contains the code for the ML inference API that makes the predictions.

## Run locally
Run the following from `src/api`:
```bash 
uv run python -m fastapi dev main.py
```

Navigate to `http://127.0.0.1:8000#docs` to open the Swagger API documentation.

## Deploy to production
Download artifacts before image build:
```bash
python -m api.scripts.download_inference_artifacts --tracking-uri <MLFLOW_TRACKING_URI> --model-type <MODEL_TYPE> --model-alias production --final-models-experiment-id <FINAL_MODELS_EXPERIMENT_ID> --model-selection-experiment-id <MODEL_SELECTION_EXPERIMENT_ID>
```
The artifact download script writes these files to `src/api/artifacts/<model_type>` by default:
- `model/`
- `row_wise_features.pkl`
- `column_wise_features.pkl`
- `fitted_column_transformer.pkl`
- `params.json`
- `best_params.json`
- `best_features.json`
- `selected_features.json`
- `categorical_mapping.json`
- `manifest.json`

Then build the SHAP background sample (200 random training passes, written to `shap_background.csv` in each model folder) from `src/`:
```bash
python -m api.scripts.build_shap_background
```

Run the Github Actions workflow defined in `.github\workflows\deploy-api.yml`

## Run in Docker
Build Docker image from root directory
```bash
docker build -t football-analytics-api:latest -f src/api/Dockerfile .   
```
Run API container:
```bash
docker run --rm -p 8000:7860 football-analytics-api:latest
```
Navigate to `http://localhost:8000/docs` to open the Swagger documentation