# test_api.py - скрипт для тестирования API
import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    response = requests.get(f"{API_URL}/health")
    print(f"Health check: {response.json()}")

def test_predict():
    data = {
        "days_overdue": 120,
        "debt_amount": 1500000,
        "payment_history_score": 45,
        "debt_to_contract_ratio": 0.75,
        "contract_amount": 2000000,
        "late_fee_amount": 75000,
        "payment_terms_days": 30,
        "annual_revenue_per_employee": 2500000,
        "employees": 50,
        "years_in_market": 5
    }

    response = requests.post(
        f"{API_URL}/predict",
        headers={"Content-Type": "application/json"},
        json=data
    )
    print(f"Prediction: {response.json()}")

def test_batch():
    data = {
        "companies": [
            {"company_id": 1, "company_name": "Company A", "days_overdue": 120, "debt_amount": 1500000},
            {"company_id": 2, "company_name": "Company B", "days_overdue": 0, "debt_amount": 0}
        ]
    }

    response = requests.post(
        f"{API_URL}/predict_batch",
        headers={"Content-Type": "application/json"},
        json=data
    )
    print(f"Batch prediction: {response.json()}")

if __name__ == "__main__":
    test_health()
    test_predict()
    test_batch()
