import os
import sys

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils.calculate_statistics import preprocess_data_all
from app.utils.forecast_utils import evaluate_model

from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.validator.bake_request_validator import PayloadStatsRequest

from app.models.schemas import BakeMeasurement
import app.services.anomaly_service as anomaly_service

# Example DataFrame for testing
data = {
    'Timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'Phase_1': np.random.rand(10),
    'Phase_2': np.random.rand(10),
    'Phase_3': np.random.rand(10)
}

df = pd.DataFrame(data)
df['Total_Baking_Time'] = df[['Phase_1', 'Phase_2', 'Phase_3']].sum(axis=1)

def test_preprocess_data_all():
    processed_df = preprocess_data_all(df.copy())
    assert "Total_Baking_Time" in processed_df.columns
    assert processed_df["Total_Baking_Time"].isnull().sum() == 0



def test_evaluate_model_daily():

    test_data = {'Total_Baking_Time': np.array([10, 20, 30, 40, 50])}
    forecast_data = np.array([12, 18, 33, 37, 55])
    test_df = pd.DataFrame(test_data)

    item = PayloadStatsRequest(file_name="dummy.csv", daily=True, test_size=0.5, outliers=False)

    # Call the function
    result = evaluate_model("Test method", forecast_data, test_df, item)

    # Assert expected results
    expected_mae = mean_absolute_error(test_df['Total_Baking_Time'], forecast_data)
    expected_rmse = np.sqrt(mean_squared_error(test_df['Total_Baking_Time'], forecast_data))
    expected_mape = np.mean(np.abs((test_df['Total_Baking_Time'] - forecast_data) / test_df['Total_Baking_Time'])) * 100
    expected_accuracy = 100 - expected_mape

    assert result["method"] == "Test method"
    assert np.isclose(result["RMSE"], expected_rmse)
    assert np.isclose(result["Mape"], expected_mape)
    assert np.isclose(result["Accuracy"], expected_accuracy)

def test_evaluate_model_non_daily():

    """
    Test function for evaluate_model with daily=False.

    Checks:
    - RMSE and MAE calculations match expected values.
    """
    test_data = {'Total_Baking_Time': np.array([10, 20, 30, 40, 50])}
    forecast_data = np.array([12, 18, 33, 37, 55])
    test_df = pd.DataFrame(test_data)

    item = PayloadStatsRequest(file_name="dummy.csv", daily=False, test_size=0.5, outliers=False)

    # Call the function
    result = evaluate_model("Test method", forecast_data, test_df, item)

    # Assert expected results
    expected_mae = mean_absolute_error(test_df['Total_Baking_Time'], forecast_data)
    expected_rmse = np.sqrt(mean_squared_error(test_df['Total_Baking_Time'], forecast_data))

    assert result["method"] == "Test method"
    assert np.isclose(result["RMSE"], expected_rmse)
    assert np.isclose(result["MAE"], expected_mae)

def test_detect_anomalies_single(monkeypatch):

    """
    Test function for detect_anomalies_single.

    Checks:
    - Whether anomalies are detected correctly based on mock input and statistics.
    """

    measurement_data = {
    "Phase_01": 930.85,
    "Phase_02": 155.43,
    "Phase_03": 107.50,
    "Phase_04": 60.20,
    "Phase_05": 19.02,
    "Phase_06": 100.50,
    "Phase_07": 34.11,
    "Phase_08": 26.50,
    "Phase_09": 4.58,
    "Cake_ID": "abc123",
    "Oven": "Oven_1"
    }

    measurement = BakeMeasurement(measurement_data)

    fake_stats = {f"Phase_0{i}": {"mean": 0.0, "std": 1.0} for i in range(1, 10)}
    monkeypatch.setattr(anomaly_service, "load_statistics", lambda: fake_stats)
    warnings = anomaly_service.detect_anomalies_single(measurement)

    # Expected warnings based on the mock data and statistics
    assert "Phase_04" in warnings
    assert "Phase_06" in warnings
