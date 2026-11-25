"""API tests: call the endpoints the way the frontend will."""
from rest_framework.test import APIClient

client = APIClient()

CIVIC = {
    "make": "Honda",
    "model": "Civic",
    "year": 2018,
    "mileage": 60000,
    "engine_cc": 1800,
    "fuel": "Petrol",
    "transmission": "Automatic",
    "city": "Lahore",
}


def test_home_page_shows_the_form():
    response = client.get("/")
    assert response.status_code == 200
    assert b"Estimate price" in response.content


def test_health_returns_ok():
    response = client.get("/api/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_options_lists_makes_and_models():
    makes = client.get("/api/options/").json()["makes"]
    assert "Civic" in makes["Honda"]


def test_predict_returns_price_inside_its_range():
    response = client.post("/api/predict/", CIVIC, format="json")
    assert response.status_code == 200
    body = response.json()
    assert body["price_range"]["low"] < body["predicted_price"] < body["price_range"]["high"]
    assert len(body["similar_listings"]) > 0


def test_predict_ignores_letter_case_and_unknown_city():
    car = {**CIVIC, "make": "HONDA", "model": "civic", "city": "Some Village"}
    assert client.post("/api/predict/", car, format="json").status_code == 200


def test_predict_rejects_model_from_another_make():
    car = {**CIVIC, "model": "Corolla"}
    response = client.post("/api/predict/", car, format="json")
    assert response.status_code == 400
    assert "model" in response.json()


def test_predict_rejects_impossible_year():
    response = client.post("/api/predict/", {**CIVIC, "year": 1850}, format="json")
    assert response.status_code == 400


def test_predict_rejects_zero_engine_for_petrol_car():
    response = client.post("/api/predict/", {**CIVIC, "engine_cc": 0}, format="json")
    assert response.status_code == 400
    assert "engine_cc" in response.json()


def test_predict_accepts_zero_engine_for_electric_car():
    car = {**CIVIC, "make": "Tesla", "model": "Model 3", "year": 2021,
           "engine_cc": 0, "fuel": "Electric"}
    assert client.post("/api/predict/", car, format="json").status_code == 200


def test_common_car_is_not_flagged_as_low_data():
    body = client.post("/api/predict/", CIVIC, format="json").json()
    assert body["model_listings"] >= 30 and body["low_data"] is False


def test_rare_car_gets_less_reliable_warning():
    levin = {**CIVIC, "make": "Toyota", "model": "Levin", "engine_cc": 1600}
    body = client.post("/api/predict/", levin, format="json").json()
    assert body["low_data"] is True and body["model_listings"] < 30
