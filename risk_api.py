# =====================================================
# API ДЛЯ ПРОГНОЗИРОВАНИЯ РИСКА ДЕФОЛТА
# Запуск: python risk_api.py
# =====================================================

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)

# Глобальные переменные для модели и признаков
model = None
FEATURES = None

def load_model():
    """Загрузка модели при старте"""
    global model, FEATURES
    
    # Загрузка модели
    if os.path.exists('catboost_risk_model.pkl'):
        model = joblib.load('catboost_risk_model.pkl')
        print("Модель загружена из catboost_risk_model.pkl")
    else:
        print("ОШИБКА: Файл модели не найден")
        return False
    
    # Загрузка метаданных с признаками
    if os.path.exists('model_metadata.json'):
        with open('model_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            FEATURES = metadata.get('features', [
                'days_overdue', 'debt_amount', 'payment_history_score',
                'debt_to_contract_ratio', 'contract_amount', 'late_fee_amount',
                'payment_terms_days', 'annual_revenue_per_employee',
                'employees', 'years_in_market'
            ])
        print(f"Загружено {len(FEATURES)} признаков из metadata")
    else:
        FEATURES = [
            'days_overdue', 'debt_amount', 'payment_history_score',
            'debt_to_contract_ratio', 'contract_amount', 'late_fee_amount',
            'payment_terms_days', 'annual_revenue_per_employee',
            'employees', 'years_in_market'
        ]
        print(f"Используются стандартные признаки: {len(FEATURES)}")
    
    return True

@app.route('/health', methods=['GET'])
def health():
    """Проверка работоспособности API"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Предсказание риска дефолта по признакам компании"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Извлечение признаков
        features = []
        missing_features = []
        
        for f in FEATURES:
            value = data.get(f)
            if value is None:
                missing_features.append(f)
                value = 0
            features.append(value)
        
        # Предсказание
        X = pd.DataFrame([features], columns=FEATURES)
        probability = model.predict_proba(X)[0, 1]
        prediction = 1 if probability > 0.5 else 0
        
        # Определение уровня риска
        if probability > 0.7:
            risk_level = "Высокий"
        elif probability > 0.3:
            risk_level = "Средний"
        else:
            risk_level = "Низкий"
        
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'missing_features': missing_features
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Пакетное предсказание для нескольких компаний"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        
        if not data or 'companies' not in data:
            return jsonify({'error': 'No companies data provided'}), 400
        
        companies = data['companies']
        results = []
        
        for company in companies:
            features = [company.get(f, 0) for f in FEATURES]
            X = pd.DataFrame([features], columns=FEATURES)
            probability = model.predict_proba(X)[0, 1]
            prediction = 1 if probability > 0.5 else 0
            
            risk_level = "Высокий" if probability > 0.7 else ("Средний" if probability > 0.3 else "Низкий")
            
            results.append({
                'company_id': company.get('company_id', 'unknown'),
                'company_name': company.get('company_name', 'unknown'),
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level
            })
        
        return jsonify({
            'total': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Получение списка признаков, используемых моделью"""
    return jsonify({
        'features': FEATURES,
        'count': len(FEATURES) if FEATURES else 0
    })

if __name__ == '__main__':
    # Загрузка модели при запуске
    if load_model():
        print("\n" + "=" * 50)
        print("API сервер запущен")
        print("=" * 50)
        print(f"URL: http://localhost:5000")
        print("\nДоступные эндпоинты:")
        print("  GET  /health       - проверка статуса")
        print("  GET  /features     - список признаков")
        print("  POST /predict      - предсказание для одной компании")
        print("  POST /predict_batch - пакетное предсказание")
        print("\nПример запроса:")
        print("  curl -X POST http://localhost:5000/predict \\")
        print('    -H "Content-Type: application/json" \\')
        print('    -d "{\\"days_overdue\\": 120, \\"debt_amount\\": 1500000}"')
        print("\n" + "=" * 50)
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Ошибка загрузки модели. Проверьте наличие файла catboost_risk_model.pkl")