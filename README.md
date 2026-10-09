
# Smart E-Learning Assistant API

Academic Performance Prediction using Machine Learning and OULAD.

## Requirements

Python 3.10+

## Installation

From the project root:

```bash
python -m pip install -r api/requirements.txt
```

## Run the API

```bash
python -m uvicorn api.main:app --reload
```

## API Documentation

After starting the server, open:

http://127.0.0.1:8000/docs

## Endpoints

- `GET /` — API information
- `GET /health` — Health check
- `POST /predict` — Predict academic performance

## Model

The API loads the trained model from:

`models/academic_performance_classifier.joblib`

The model was trained using OULAD data.
