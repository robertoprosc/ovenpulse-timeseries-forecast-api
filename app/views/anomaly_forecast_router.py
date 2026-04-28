import json

from fastapi import APIRouter, Response

from app.services.anomaly_service import detect_anomalies_single,scan_for_daily_trends
from app.utils.calculate_statistics import plot_trends,calculate_cov_and_select, preprocess_data
from app.utils.forecast_utils import *
from config import Settings

from common.logger import logging_setup


from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from pylab import rcParams


# **********************
# IMPORT PROJECT MODULES
# **********************

from app.validator.bake_request_validator import PayloadAnomalyRequest,PayloadStatsRequest
from app.models.schemas import BakeMeasurement

# Logger configuration
logging_setup.setup_logging()
logger = logging.getLogger()

settings = Settings()

# defining viewname
router = APIRouter()


@router.get('/')
async def healthcheck():
    return Response(json.dumps({'Status': 'Available'}),
                    status_code=200,
                    media_type='application/json')

@router.post('/anomaly_detection_single')
async def anomaly_detection_single(item: PayloadAnomalyRequest):
    """
    This function manages the "anomaly_detection" for a single request containing a single measurement

    Parameters
    ----------
    item: PayloadAnomalyRequest
        PayloadAnomalyRequest object for mapping the incoming Measurement request and validating it

    Returns
    -------
    response_json : Response
        JSON response containing warnings about a particular anomaly-phase
    """

    try:

        measurement = BakeMeasurement(item.single_measurement)

        warnings = detect_anomalies_single(measurement)

        if warnings is None:
            response = Response(json.dumps({'error': 'Prediction service error'}),
                                status_code=500, media_type='application/json')
        else:
            if not warnings:
                response = Response(json.dumps(dict(status="ok",message="No Anomalies detected")), media_type='application/json')
            else:
                response = Response(json.dumps(warnings), media_type='application/json')
    except Exception as e:
        logger.error('PREDICTION ERROR', exc_info=e)
        response = Response(json.dumps({'error': 'Prediction service error'}),
                            status_code=500, media_type='application/json')
    return response

@router.post('/anomaly_detection_massive')
async def anomaly_detection_massive(item: PayloadAnomalyRequest):
    """
    This function manages the "anomaly_detection_massive" relative to a specific dataset with statistics returned

    Parameters
    ----------
    item: PayloadAnomalyRequest
        PayloadAnomalyRequest object for mapping the incoming Measurement request and validating it

    Returns
    -------
    response_json : Response
        JSON response containing warnings about a particular dataset
        example:

        {
            "status": "ok",
            "message": "932/1820 - Anomalies detected",
            "percentage": "51.21%"
        }

    """

    try:

        test_dataset=os.path.join(settings.RESOURCES_PATH,"dataset","test_dataset",item.test_set)
        df=pd.read_excel(test_dataset)

        warnings_list=[]

        for idx,row in df.iterrows():

            measurement = BakeMeasurement(row[1:-2].to_dict())

            warnings = detect_anomalies_single(measurement)

            if warnings:
                warnings_list.append(warnings)

        percent_anomalies = (len(warnings_list) / len(df)) * 100

        if warnings is None:
            response = Response(json.dumps({'error': 'Prediction service error'}),
                                status_code=500, media_type='application/json')
        else:
            if not warnings:
                response = Response(json.dumps(dict(status="ok",message="No Anomalies detected")), media_type='application/json')
            else:
                results=dict(status="ok",
                             message=f"{len(warnings_list)}/{len(df)} - Anomalies detected",
                             percentage=f"{percent_anomalies:.2f}%"
                             )
                response = Response(json.dumps(results), media_type='application/json')
    except Exception as e:
        logger.error('PREDICTION ERROR', exc_info=e)
        response = Response(json.dumps({'error': 'Prediction service error'}),
                            status_code=500, media_type='application/json')
    return response


@router.post('/anomaly_trend_detection')
async def anomaly_trend_detection(item: PayloadAnomalyRequest):
    """
    This function manages the "anomaly_trend_detection" relative to a specific dataset, return alert using a logger object

    Parameters
    ----------
    item: PayloadAnomalyRequest
        PayloadAnomalyRequest object for mapping the incoming Measurement request and validating it

    Returns
    -------
    response_json : Response
        JSON response containing stats warnings about a particular dataset and trend increasing detected
        example:


    """

    try:

        test_dataset = os.path.join(settings.RESOURCES_PATH, "dataset", "train_dataset", item.test_set)
        df = pd.read_excel(test_dataset)

        # Scan for trends and generate alerts
        daily_alerts = scan_for_daily_trends(df)
        for date, sub_phases in daily_alerts.items():
            for sub_phase, message in sub_phases.items():
                logging.info(f"Date: {date}, Sub-Phase: {sub_phase}, Alert: {message}")


        plot_trends(df, daily_alerts,settings.TREND_DETECTION_GRAPHS)

        warnings={}
        if warnings is None:
            response = Response(json.dumps({'error': 'Prediction service error'}),
                                status_code=500, media_type='application/json')
        else:
            if not warnings:
                response = Response(json.dumps(dict(status="ok", message="No Anomalies detected")),
                                    media_type='application/json')
            else:
                results = dict(status="ok",graphs_path=settings.TREND_DETECTION_GRAPHS)
                response = Response(json.dumps(results), media_type='application/json')
    except Exception as e:
        logger.error('PREDICTION ERROR', exc_info=e)
        response = Response(json.dumps({'error': 'Prediction service error'}),
                            status_code=500, media_type='application/json')
    return response

@router.post('/forecasting_bakery_time')
async def get_forecasting_bakery_time(item : PayloadStatsRequest):
    """
    This function manages the "forecasting_bakery_time" prediction service for overall days and save
    the plots in a specific directory.

     Parameters
    ----------
    item: PayloadAnomalyRequest
        PayloadAnomalyRequest object for mapping the incoming Measurement request and validating it

    Parameters

    response_json : Response
        JSON response for prediction outcome
    """

    try:

        dataset_path = item.file_name  # Update this to the actual path of your dataset
        df = pd.read_csv(dataset_path)

        if item.daily:
            # Preprocess the dataset
            df = preprocess_data(df)

            df = df.groupby('Date')['Total_Baking_Time'].mean().reset_index()

            # Ensure the data is sorted by date
            df.sort_values('Date', inplace=True)
            df.set_index('Date', inplace=True)
        else:
            df = preprocess_data(df)

            # Ensure the data is sorted by timestamp
            df.sort_values('Date', inplace=True)
            df.set_index('Date', inplace=True)


        train_size = int(len(df) * 0.8)
        train, test = df.iloc[:train_size], df.iloc[train_size:]

        # Perform the ADF test on the training data
        adf_result = adfuller(train['Total_Baking_Time'])
        print('ADF Statistic:', adf_result[0])
        print('p-value:', adf_result[1])
        print('Critical Values:', adf_result[4])

        # Interpret the result
        if adf_result[1] < 0.05:
            print("The time series is stationary.")
        else:
            print("The time series is non-stationary and may require differencing.")


        #DATA IMPORT
        #---------------------------------------------------------------------#
        #MODEL IMPORT E TESTS
        # Naive Method
        # Directory to save plots
        save_dir = './resources/results/forecast_plot/'

        # Naive Method
        naive_forecast = test['Total_Baking_Time'].shift(1).fillna(method='bfill')
        naive_results = evaluate_model('Naive method', naive_forecast, test,item)
        plot_forecast(train, test, pd.DataFrame({'Forecast': naive_forecast}, index=test.index),
                      'Naive Method Forecast', save_dir, 'naive_forecast.png')

        # Simple Average Method
        avg_forecast = pd.Series([train['Total_Baking_Time'].mean()] * len(test), index=test.index)
        avg_results = evaluate_model('Simple average method', avg_forecast, test,item)
        plot_forecast(train, test, pd.DataFrame({'Forecast': avg_forecast}, index=test.index),
                      'Simple Average Method Forecast', save_dir, 'avg_forecast.png')

        # Simple Moving Average Method
        moving_avg_forecast = train['Total_Baking_Time'].rolling(window=5).mean().iloc[-1]
        moving_avg_forecast = pd.Series([moving_avg_forecast] * len(test), index=test.index)
        moving_avg_results = evaluate_model('Simple moving average forecast', moving_avg_forecast, test,item)
        plot_forecast(train, test, pd.DataFrame({'Forecast': moving_avg_forecast}, index=test.index),
                      'Simple Moving Average Forecast', save_dir, 'moving_avg_forecast.png')


        # Holt's Exponential Smoothing
        holt_model = ExponentialSmoothing(train['Total_Baking_Time'], trend='add').fit()
        holt_forecast = holt_model.forecast(steps=len(test))
        holt_results = evaluate_model("Holt's exponential smoothing method", holt_forecast, test,item)
        plot_forecast(train, test, pd.DataFrame({'Forecast': holt_forecast}, index=test.index),
                      "Holt's Exponential Smoothing Method Forecast", save_dir, 'holt_forecast.png')

        # Holt-Winters' Additive Method
        hw_add_model = ExponentialSmoothing(train['Total_Baking_Time'], seasonal_periods=7, trend='add',
                                            seasonal='add').fit()
        hw_add_forecast = hw_add_model.forecast(len(test))
        hw_add_results = evaluate_model("Holt Winters' additive method", hw_add_forecast, test,item)
        plot_forecast(train, test, pd.DataFrame({'Forecast': hw_add_forecast}, index=test.index),
                      "Holt Winters' Additive Method Forecast", save_dir, 'hw_add_forecast.png')

        # Holt-Winters' Multiplicative Method
        hw_mul_model = ExponentialSmoothing(train['Total_Baking_Time'], seasonal_periods=7, trend='add',
                                            seasonal='mul').fit()
        hw_mul_forecast = hw_mul_model.forecast(len(test))
        hw_mul_results = evaluate_model("Holt Winters' multiplicative method", hw_mul_forecast, test,item)
        plot_forecast(train, test, pd.DataFrame({'Forecast': hw_mul_forecast}, index=test.index),
                      "Holt Winters' Multiplicative Method Forecast", save_dir, 'hw_mul_forecast.png')

        # Autoregressive (AR) Method
        ar_model = AutoReg(train['Total_Baking_Time'], lags=1).fit()
        ar_forecast = ar_model.predict(start=len(train), end=len(train) + len(test) - 1)
        ar_results = evaluate_model("Autoregressive (AR) method", ar_forecast, test,item)
        plot_forecast(train, test, pd.DataFrame({'Forecast': ar_forecast}, index=test.index),
                      "Autoregressive (AR) Method Forecast", save_dir, 'ar_forecast.png')

        # ARIMA Method
        arima_model = ARIMA(train['Total_Baking_Time'], order=(1, 1, 1))
        arima_result = arima_model.fit()
        arima_forecast = arima_result.forecast(steps=len(test))
        arima_results = evaluate_model('ARIMA method', arima_forecast, test,item)
        plot_forecast(train, test, pd.DataFrame({'Forecast': arima_forecast}, index=test.index),
                      'ARIMA Method Forecast', save_dir, 'arima_forecast.png')

        # SARIMA Method
        sarima_model = SARIMAX(train['Total_Baking_Time'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        sarima_result = sarima_model.fit(disp=False)
        sarima_forecast = sarima_result.get_forecast(steps=len(test))
        sarima_forecast_values = sarima_forecast.predicted_mean
        sarima_forecast_conf_int = sarima_forecast.conf_int()
        sarima_forecast_df = pd.DataFrame({'Forecast': sarima_forecast_values}, index=test.index)
        sarima_results = evaluate_model('SARIMA method', sarima_forecast_values, test,item)
        plot_forecast(train, test, sarima_forecast_df, 'SARIMA Method Forecast', save_dir, 'sarima_forecast.png',
                      sarima_forecast_conf_int)

        # Prophet Method
        prophet_train = train.reset_index().rename(columns={'Date': 'ds', 'Total_Baking_Time': 'y'})
        prophet_test = test.reset_index().rename(columns={'Date': 'ds', 'Total_Baking_Time': 'y'})

        prophet_model = Prophet()
        prophet_model.fit(prophet_train)

        prophet_future = prophet_model.make_future_dataframe(periods=len(test))
        prophet_forecast = prophet_model.predict(prophet_future)
        prophet_forecast = prophet_forecast.set_index('ds')['yhat'][-len(test):]

        prophet_results = evaluate_model('Prophet method', prophet_forecast, test,item)
        plot_forecast(train, test, pd.DataFrame({'Forecast': prophet_forecast.values}, index=test.index),
                      'Prophet Method Forecast', save_dir, 'prophet_forecast.png')

        # Combine results
        results = [naive_results, avg_results, moving_avg_results,
                   holt_results, hw_add_results, hw_mul_results, ar_results, arima_results, sarima_results,
                   prophet_results]

        results_df = pd.DataFrame(results)
        print(results_df)

        response_dict=results
        if response_dict is None:
            response = Response(json.dumps({'error': 'Prediction service error'}),
                                status_code=500, media_type='application/json')
        else:
            response = Response(json.dumps(response_dict), media_type='application/json')
    except Exception as e:
        logger.error('PREDICTION ERROR', exc_info=e)
        response = Response(json.dumps({'error': 'Prediction service error'}),
                            status_code=500, media_type='application/json')
    return response


@router.post('/forecasting_bakery_time_phases')
async def get_forecasting_bakery_time(item : PayloadStatsRequest):
    """
    This function manages the "forecasting_bakery_time" prediction service for overall days in specifc sub_phases and save
    the plots in a specific directory.

     Parameters
    ----------
    item: PayloadAnomalyRequest
        PayloadAnomalyRequest object for mapping the incoming Measurement request and validating it

    Parameters

    response_json : Response
        JSON response for prediction outcome
    """


    try:
        save_dir = './resources/results/forecast_plot_phase/'
        os.makedirs(save_dir, exist_ok=True)

        dataset_path = item.file_name  # Update this to the actual path of your dataset
        df = pd.read_csv(dataset_path)

        if item.daily:
            # Preprocess the dataset
            df = preprocess_data(df)

            phase_columns = [i for i in df.columns if i.startswith('Phase_')]
            # Calculate CoV and select the best sub-phase
            best_sub_phase, cov_df = calculate_cov_and_select(df, phase_columns)
            cov_df.sort_values(by=['Coeff_of_Var'],inplace=True)
            print(f"Best sub-phase to analyze: {best_sub_phase}")

            df_daily = df.groupby('Date')['Total_Baking_Time'].mean().reset_index()
            # Ensure the data is sorted by date
            df_daily.sort_values('Date', inplace=True)
            df_daily.set_index('Date', inplace=True)
        else:
            df = preprocess_data(df)

            # we calculate the best sub_phases with least Coefficient of Variation.
            phase_columns = [i for i in df.columns if i.startswith('Phase_')]
            # Calculate CoV and select the best sub-phase
            best_sub_phase, cov_df = calculate_cov_and_select(df, phase_columns)
            cov_df.sort_values(by=['Coeff_of_Var'])
            print(f"Best sub-phase to analyze: {best_sub_phase}")

            # Ensure the data is sorted by timestamp
            df.sort_values('Date', inplace=True)
            df.set_index('Date', inplace=True)


        # Initialize results container
        results = []

        #SEASONAL DECOMPOSITION - ADDITIVE
        rcParams['figure.figsize'] = 12, 8
        decomposition = seasonal_decompose(df_daily['Total_Baking_Time'], model='additive',period=7)
        fig = decomposition.plot()
        plt.title('Seasonal Decomposition of Total Baking Time - Additive')
        plt.savefig(os.path.join(save_dir, 'seasonal_decomposition_additive.png'))
        plt.close()

        #SEASONAL DECOMPOSITION - MULTIPLICATIVE
        rcParams['figure.figsize'] = 12, 8
        decomposition = seasonal_decompose(df_daily['Total_Baking_Time'], model='multiplicative',period=7)
        fig = decomposition.plot()
        plt.title('Seasonal Decomposition of Total Baking Time - Multiplicative')
        plt.savefig(os.path.join(save_dir, 'seasonal_decomposition_multiplicative.png'))
        plt.close()

        #we choose the best sub_phases with least Coefficient of Variation. It means Total Bakery Time for these Phases have been consistent over the train set period
        #I choose the first 4 values
        cov_df_sub_phases=cov_df['Sub_Phase'][:4].values

        test_size=0.8
        # Loop over each sub-phase column
        for sub_phase in cov_df_sub_phases:
            print(f"Forecasting for {sub_phase}...")

            df_sub_phase = df.groupby('Date')[sub_phase].mean().reset_index()
            # Group by date and aggregate sub-phase values



            train_size = int(len(df_sub_phase) * 0.8)
            train, test = df_sub_phase.iloc[:train_size], df_sub_phase.iloc[train_size:]

            # SARIMA parameters (you may need to tune these)
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 7)  # Example seasonal order with weekly seasonality

            # Fit SARIMA model
            sarima_model = SARIMAX(train[sub_phase], order=order, seasonal_order=seasonal_order)
            sarima_result = sarima_model.fit(disp=False)

            # Forecast
            forecast_steps = len(test)
            sarima_forecast = sarima_result.get_forecast(steps=forecast_steps)
            sarima_forecast_values = sarima_forecast.predicted_mean
            sarima_forecast_conf_int = sarima_forecast.conf_int()

            mse = mean_squared_error(test[sub_phase], sarima_forecast_values)
            mape = calculate_mape(test[sub_phase], sarima_forecast_values)
            mae = mean_absolute_error(test[sub_phase], sarima_forecast_values)
            rmse = np.sqrt(mse)
            accuracy = 100 - mape

            print(f"MSE for {sub_phase}: {mse}")
            print(f"MAPE for {sub_phase}: {mape}%")
            print(f"MAE for {sub_phase}: {mae}")
            print(f"RMSE for {sub_phase}: {rmse}")
            print(f"Accuracy for {sub_phase}: {accuracy}")

            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(train.index, train[sub_phase], label='Training Data')
            plt.plot(test.index, test[sub_phase], label='Test Data')
            plt.plot(sarima_forecast_values.index, sarima_forecast_values, label='Forecast', linestyle='--')
            plt.fill_between(sarima_forecast_values.index, sarima_forecast_conf_int.iloc[:, 0],
                             sarima_forecast_conf_int.iloc[:, 1], color='k', alpha=0.2)
            plt.title(f'SARIMA Forecast for {sub_phase}')
            plt.xlabel('Date')
            plt.ylabel(f'{sub_phase} Value')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'sarima_forecast_{sub_phase}.png'))
            plt.close()

            # Collect results if needed
            results.append({
                'sub_phase': sub_phase,
                'forecast_values': sarima_forecast_values,
                'forecast_conf_int': sarima_forecast_conf_int,
                'mse': mse,
                'mape': mape,
                'mae': mae,
                'rmse': rmse,
                'accuracy': accuracy
            })
        return results

        results_df = pd.DataFrame(results)
        print(results_df)

        response_dict=results
        if response_dict is None:
            response = Response(json.dumps({'error': 'Prediction service error'}),
                                status_code=500, media_type='application/json')
        else:
            response = Response(json.dumps(response_dict), media_type='application/json')
    except Exception as e:
        logger.error('PREDICTION ERROR', exc_info=e)
        response = Response(json.dumps({'error': 'Prediction service error'}),
                            status_code=500, media_type='application/json')
    return response