# run_mlflow_server.bat - Запуск MLflow сервера для Windows
@echo off
set MLFLOW_TRACKING_URI=file:./mlruns
mlflow models serve -m "models:/CatBoost_Default_Risk_Classifier/1" -p 5001 --no-conda
pause
