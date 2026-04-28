import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

import logging

from common.logger import logging_setup

# Logger configuration
logging_setup.setup_logging()
logger = logging.getLogger()


def calculate_rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast)**2))

# Function to calculate MAPE
def calculate_mape(true_values, forecast_values):
    # Ensure forecast_values is a Series with the same index as true_values
    forecast_values = pd.Series(forecast_values, index=true_values.index)

    # Calculate MAPE
    mape = np.mean(np.abs((true_values - forecast_values) / true_values)) * 100
    return mape

def evaluate_model(method, forecast,test,item):
    mae = mean_absolute_error(test['Total_Baking_Time'], forecast)
    mse = mean_squared_error(test['Total_Baking_Time'], forecast)
    rmse = np.sqrt(mse)
    if item.daily:
        mape = calculate_mape(test['Total_Baking_Time'], forecast)
        accuracy = 100 - mape
        results = {'method': method, 'RMSE': rmse, 'Mape': mape,'Accuracy':accuracy}
    else:
        results={'method': method, 'RMSE': rmse, 'MAE': mae}
    logger.info(results)
    return results


def plot_forecast(train, test, forecast_df, title, save_dir, filename, conf_int=None):
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['Total_Baking_Time'], label='Training Data')
    plt.plot(test.index, test['Total_Baking_Time'], label='Test Data')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', linestyle='--')

    if conf_int is not None:
        plt.fill_between(forecast_df.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='k', alpha=0.2)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Total Baking Time')
    plt.legend()

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def plot_sarima(train,test,sarima_forecast_values,sarima_forecast,save_dir,filename):
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['Total_Baking_Time'], label='Training Data')
    plt.plot(test.index, test['Total_Baking_Time'], label='Test Data')
    plt.plot(test.index, sarima_forecast_values, label='SARIMA Forecast', linestyle='--')
    plt.fill_between(test.index, sarima_forecast.conf_int().iloc[:, 0], sarima_forecast.conf_int().iloc[:, 1],
                     color='k', alpha=0.2)
    plt.title('Total Baking Time Forecast')
    plt.xlabel('Date')
    plt.ylabel('Total Baking Time')
    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
