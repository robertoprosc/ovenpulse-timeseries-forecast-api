import os
import logging
from common.logger import logging_setup

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from config import Settings

from sklearn.model_selection import train_test_split

# Logger configuration
logging_setup.setup_logging()
logger = logging.getLogger()

settings = Settings()

def calculate_statistics_batch(df):
    """
    Calculate mean and std of every phase in the dataset.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the measurements.

    Returns
    -------
    statistics: dict
        Dictionary containing mean and std for each phase.
        Example: {'Phase_01': {'mean': 10.0, 'std': 2.5}, ...}
    """
    statistics = {}

    for phase in df.filter(like='Phase').columns:
        phase_stats = {
            'mean': df[phase].mean(),
            'std': df[phase].std()
        }
        statistics[phase] = phase_stats

    return statistics


def calculate_affecting_phase(df):
    """
    Calculate the Phase that affects the backing time using correlation between Total Baking Time and every Phase

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the measurements.

    Returns
    -------
    statistics: dict
        Dictionary containing Phase_name and correlation grade that affects baking time
        Example: {'Phase_name': 'Phase_01,'Correlation': 0.85'}, ...}
    """

    phase_columns = [col for col in df.columns if col.startswith('Phase')]
    df_modified = df.copy()
    df_modified['Total_Baking_Time'] = df_modified[phase_columns].sum(axis=1)

    phase_impacts = df_modified.corr()['Total_Baking_Time'].sort_values(ascending=False)
    logger.info("Phases affecting baking time the most:")

    first_phase_name = phase_impacts.index[1]  # Access the first index (phase name)
    first_phase_value = phase_impacts.iloc[1]  # Access the first value (correlation)

    logger.info(f"First Phase: {first_phase_name}, Correlation: {first_phase_value:.4f}")

    return dict(Phase_name=first_phase_name,correlation=first_phase_value)


def plot_trends(df, daily_alerts, output_dir='plots/'):
    """
    Plots trends and anomalies detected in the data.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the measurements.
    daily_alerts: dict
        Dictionary containing alerts for each day and sub-phase with increasing trends.
    output_dir: str
        Directory path where plots will be saved.

    Returns
    -------
    None
    """
    sub_phases = [col for col in df.columns if col.startswith('Phase_')]

    for sub_phase in sub_phases:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Timestamp', y=sub_phase, data=df, hue='Date', palette='viridis')
        plt.title(f'{sub_phase} Trend and Anomalies')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')

        # Highlight days with anomalies
        for date, alerts in daily_alerts.items():
            if sub_phase in alerts:
                anomaly_data = df[(df['Date'] == date) & (df[sub_phase].notna())]
                plt.scatter(anomaly_data['Timestamp'], anomaly_data[sub_phase], color='red', label='Anomaly')

        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sub_phase}_trend.png')  # Save each plot as an image
        plt.close()


def preprocess_data(df):
    """
    Preprocess the dataset by removing null values and calculating total baking time by day.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the measurements.

    Returns
    -------
    df_daily : pd.DataFrame
        DataFrame grouped by day with total baking time.
    """
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Convert timestamp column to datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Ensure all phase columns contain float64 data
    phase_columns = [i for i in df.columns if i.startswith('Phase_')]
    df[phase_columns] = df[phase_columns].astype('float64')

    # Calculate Total_Baking_Time
    df['Total_Baking_Time'] = df[phase_columns].sum(axis=1)

    # Group by Date
    df['Date'] = df['Timestamp'].dt.date


    return df


def preprocess_data_all(df):
    # Drop NaN rows if needed
    df.dropna(inplace=True)

    # Convert timestamp column to datetime if not already
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Extract date from Timestamp
    df['Date'] = df['Timestamp'].dt.date

    # Ensure all phase columns contain float64 data
    phase_columns = [col for col in df.columns if col.startswith('Phase_')]
    df[phase_columns] = df[phase_columns].astype('float64')

    # Calculate Total_Baking_Time
    df['Total_Baking_Time'] = df[phase_columns].sum(axis=1)

    return df

def calculate_cov_and_select(df, phase_columns):
    cov_dict = {'Mean': [], 'Std_Dev': [], 'Coeff_of_Var': [], 'Sub_Phase': []}
    for phase in phase_columns:
        train_mean = np.mean(df[phase])
        train_std = np.std(df[phase])
        train_cov = train_std / train_mean
        cov_dict['Mean'].append(train_mean)
        cov_dict['Std_Dev'].append(train_std)
        cov_dict['Coeff_of_Var'].append(train_cov)
        cov_dict['Sub_Phase'].append(phase)
    cov_df = pd.DataFrame(cov_dict)
    best_sub_phase = cov_df.loc[cov_df['Coeff_of_Var'].idxmin()]['Sub_Phase']
    return best_sub_phase, cov_df