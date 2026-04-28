
from app.utils.load_data import load_statistics
from app.models.schemas import BakeMeasurement

import pandas as pd
import numpy as np
from scipy.stats import linregress,kendalltau
import logging

from common.logger import logging_setup
import logging

# Logger configuration
logging_setup.setup_logging()
logger = logging.getLogger()

def detect_anomalies_single(data : BakeMeasurement):

    stats_values = load_statistics()
    anomaly_threshold = 3

    warnings = {}
    for feature, value in data.to_dict().items():
        if feature.startswith("Phase_"):
            if abs(value - stats_values[feature]["mean"]) > anomaly_threshold * stats_values[feature]["std"]:
                logging.info(f"Feature : {feature} contains anomalies - Cake : {data.cake_id} - Oven: {data.oven}")
                warnings[feature] = f"contains anomalies - Cake : {data.cake_id} - Oven: {data.oven}"

    return warnings


def detect_increasing_trend(data,alpha=0.05, threshold=0.05):
    """
    Detects increasing trends in the given data using kendalltau metrics.

    Parameters
    ----------
    data: pandas.Series
        Time series data of a sub-phase.
    threshold: float
        Significance level to detect trend (default is 0.05).

    Returns
    -------
    bool
        True if an increasing trend is detected, False otherwise.
    """
    tau, p_value = kendalltau(np.arange(len(data)), data)
    return tau > 0 and p_value < alpha

# Function to scan each machine's data for increasing trends
def scan_for_trends(df):
    """
    Scans each machine's data and sends an alert if an increasing trend is detected.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the measurements.

    Returns
    -------
    alerts : dict
        Dictionary containing alerts for each machine and sub-phase with increasing trends.
    """
    alerts = {}
    sub_phases = [col for col in df.columns if col.startswith('Phase_')]

    for oven in df['Oven'].unique():
        oven_data = df[df['Oven'] == oven]
        alerts[oven] = {}

        for sub_phase in sub_phases:
            if detect_increasing_trend(oven_data[sub_phase]):
                alerts[oven][sub_phase] = "Increasing trend detected"

    return alerts



def scan_for_daily_trends(df):
    """
    Scans each day's data and sends an alert if an increasing trend is detected.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the measurements.

    Returns
    -------
    daily_alerts : dict
        Dictionary containing alerts for each day and sub-phase with increasing trends.
    """
    daily_alerts = {}
    sub_phases = [col for col in df.columns if col.startswith('Phase_')]

    for date in df['Date'].unique():
        date_data = df[df['Date'] == date]
        daily_alerts[date] = {}

        for sub_phase in sub_phases:
            if detect_increasing_trend(date_data[sub_phase]):
                daily_alerts[date][sub_phase] = "Increasing trend detected"

    return daily_alerts