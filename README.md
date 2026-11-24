# PakWheels Used Car Price Predictor

Estimate the fair market price of a used car in Pakistan from its make, model, year, mileage and more.

> Live demo, metrics and screenshots will be added as the project is built.

## Project structure

```
data/raw/        original dataset (never edited)
data/processed/  cleaned dataset
notebooks/       exploration only
src/             production ML code
models/          saved trained model
api/             Django REST API
frontend/        Streamlit app
tests/           automated tests
```

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```

`requirements.txt` holds only what the live app needs; `requirements-dev.txt` adds charting and testing tools.

Author: Shayan Ali Shah
