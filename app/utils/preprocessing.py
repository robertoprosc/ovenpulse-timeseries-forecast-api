import pandas as pd

import logging
import time

from common.logger import logging_setup

# Logger configuration
logging_setup.setup_logging()
logger = logging.getLogger()

def preprocess_data_outliers(df, test_size=0, random_state=None, phases_to_check=[]):
  """
  Preprocesses data by handling missing values, outliers (using IQR for specific phases), and optionally splitting into training and testing sets.

  Parameters:
      df (pd.DataFrame): DataFrame containing measurements.
      test_size (float, optional): Proportion of data for testing set (default 0).
      random_state (int, optional): Random seed for shuffling data (default None).
      phases_to_check (list, optional): List of phase names to check for outliers (default empty).

  Returns:
      pd.DataFrame: Preprocessed DataFrame.
      pd.DataFrame (optional): Testing set DataFrame (if test_size > 0).
  """

  total_missing_values = df.isnull().sum().sum()
  logger.info(f"Total Missing Values: {total_missing_values}")
  time.sleep(1)

  # Drop rows with missing values
  logger.info("Dropping null values ...")
  df.dropna(inplace=True)
  time.sleep(1)

  # Convert timestamp column to datetime if needed
  df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
  if 'Timestamp' in df.columns:
      df['Timestamp'] = pd.to_datetime(df['Timestamp'])

      # Split Timestamp into Date and Time columns
      df['Date'] = df['Timestamp'].dt.date
      df['Time'] = df['Timestamp'].dt.time

  logging.info("--------- PRE-PROCESSING STEPS - Outliers detection ---------")
  phase_cols = df.filter(like='Phase')

  # IQR-based outlier detection
  Q1 = phase_cols.quantile(0.25)
  Q3 = phase_cols.quantile(0.75)
  IQR = Q3 - Q1

  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Detect outliers only in specified phases
  outliers = (((phase_cols[phases_to_check]) < lower_bound.loc[phases_to_check]) |
             ((phase_cols[phases_to_check]) > upper_bound.loc[phases_to_check])).any(axis=1)

  total_rows = len(df)
  num_outliers = outliers.sum()

  # Calculate the percentage of outliers
  percent_outliers = (num_outliers / total_rows) * 100
  logging.info(f" Outliers Percentage (Phase columns - {phases_to_check}): {percent_outliers:.2f}%")

  test_set = pd.DataFrame()
  if test_size > 0:
      # Shuffle the DataFrame (important for random sampling)
      df_shuffled = df.sample(frac=1, random_state=random_state)

      # Reindex outliers to suppress warning
      outliers = outliers.reindex(df_shuffled.index)

      # Split into normal and outlier subsets
      normal_df = df_shuffled[~outliers]
      outlier_df = df_shuffled[outliers]

      # Sample equal sizes from normal and outlier subsets
      normal_test_size = int(test_size * len(outlier_df) / 2)
      outlier_test_size = int(test_size * len(outlier_df) / 2)
      normal_test_df = normal_df.sample(normal_test_size, random_state=random_state)
      outlier_test_df = outlier_df.sample(outlier_test_size, random_state=random_state)

      # Combine normal and outlier test sets
      test_set = pd.concat([normal_test_df, outlier_test_df], ignore_index=True)

  outliers = outliers.reindex(df.index)
  df = df[~outliers]
  logging.info(f" {num_outliers} outliers removed (Phase columns - {phases_to_check}).")

  if test_set.empty:
      return df
  else:
      return df, test_set



def preprocess_data(df,test_size=0,random_state=None):

    """
       data preprocessing logic here (e.g., remove null values)

       Parameters
       ----------
       df: dataFrame
           dataFrame object containing the measurements

       Returns
       -------
       df : dataFrame
           dataFrame pre-processed
       """

    total_missing_values = df.isnull().sum().sum()
    logger.info(f"Total Missing Values: {total_missing_values}")
    time.sleep(1)
    # Drop rows with missing values
    logger.info("Dropping null values ...")
    df.dropna(inplace=True)
    time.sleep(1)

    # Convert timestamp column to datetime if needed
    df.rename(columns={'Unnamed: 0' : 'Timestamp'},inplace=True)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Split Timestamp into Date and Time columns
        df['Date'] = df['Timestamp'].dt.date
        df['Time'] = df['Timestamp'].dt.time

    return df