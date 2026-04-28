# API Payload Examples

## POST /stats/preprocess_and_calculate_stats

```json
{
  "file_name": "Cakes_df.csv",
  "daily": true,
  "test_size": 0.5,
  "outliers": false
}
```

## POST /anomaly/anomaly_detection_single

```json
{
  "train_set": "train.xlsx",
  "test_set": "test.xlsx",
  "single_measurement": {
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
}
```

## POST /anomaly/forecasting_bakery_time

```json
{
  "file_name": "resources/dataset/Cakes_df.csv",
  "daily": true,
  "test_size": 0.2,
  "outliers": false
}
```

