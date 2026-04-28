import json
import os
import time
from datetime import timedelta

import logging
from fastapi import APIRouter, Response

from config import Settings

import pandas as pd

from common.logger import logging_setup

# **********************
# IMPORT PROJECT MODULES
# **********************

from app.validator.bake_request_validator import PayloadStatsRequest

from app.utils.preprocessing import preprocess_data, preprocess_data_outliers
from app.utils.calculate_statistics import calculate_statistics_batch,calculate_affecting_phase

# Logger configuration
logging_setup.setup_logging()
logger = logging.getLogger()

settings = Settings()

resource_path = settings.RESOURCES_PATH
results_path = settings.RESULTS_PATH

# defining viewname
router = APIRouter()


@router.get('/')
async def healthcheck():
    return Response(json.dumps({'Status': 'Available'}),
                    status_code=200,
                    media_type='application/json')


@router.post('/preprocess_and_calculate_stats')
async def preprocess_and_calculate_stats(item: PayloadStatsRequest):
    """
    This function manages the pre-processing and calculate statistics service for future anomaly detection

    Parameters
    ----------
    item: PayloadStatsRequest
        PayloadStatsRequest object for mapping the incoming request and validating it

    Returns
    -------
    response_json : Response
        JSON response for pre-processing and calculate-statistics status.
    """

    try:
        logger.info(f"Start pre-processing and calculate-statistics route")
        start_total = time.time()

        test_dataset_path = os.path.join(resource_path,"dataset","test_dataset")
        train_dataset_path =  os.path.join(resource_path,"dataset","train_dataset")
        stats_path = os.path.join(results_path,"stats")

        # Load dataset from specified path
        df = pd.read_csv(os.path.join(resource_path,"dataset",item.file_name))

        # Perform preprocessing and obtain test set with removed outliers
        if item.outliers:
            #REMOVING OUTLIERS FOR SPECIFIC PHASES ( HIGHER STD ) - ADAPTIVE OUTLIERS REMOVAL using z-score ?
            phases_to_check= ['Phase_03','Phase_02','Phase_06']
            df_processed, test_set = preprocess_data_outliers(df,test_size=item.test_size,phases_to_check=phases_to_check)
            test_set.to_excel(os.path.join(test_dataset_path, "test.xlsx"), index=False)
        else:
            df_processed= preprocess_data(df, test_size=item.test_size)

        # Save test set to CSV (optional)
        df_processed.to_excel(os.path.join(train_dataset_path,"train.xlsx"), index=False)

        # Initialize manager with train set and calculate statistics
        statistics = calculate_statistics_batch(df_processed)

        with open(os.path.join(stats_path,'statistics.json'), 'w') as f:
            json.dump(statistics, f,indent=4)

        end_total = time.time()

        response_dict=dict(status="ok", response_time_total=str(timedelta(seconds=end_total - start_total)),
                           test_set_path=os.path.join(test_dataset_path,"test.xlsx"),
                           train_set_path=os.path.join(train_dataset_path,"train.xlsx"),
                           statistics_path=os.path.join(stats_path,'statistics.json'))

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


@router.post('/affects_backing_time')
async def affects_backing_time(item: PayloadStatsRequest):
    """
    This function manages to output the phase that particularly affects the baking time

    Parameters
    ----------
    item: PayloadStatsRequest
        PayloadStatsRequest object for mapping the incoming request and validating it

    Returns
    -------
    response_json : Response
        JSON response for pre-processing and calculate-statistics status.
    """

    try:
        logger.info(f"Start pre-processing and calculate-statistics route")
        start_total = time.time()

        # Load dataset from specified path
        df = pd.read_csv(os.path.join(resource_path,"dataset",item.file_name))

        # Perform preprocessing and obtain test set with removed outliers
        df_processed= preprocess_data(df)

        # Initialize manager with train set and calculate statistics
        first_phase = calculate_affecting_phase(df_processed)

        end_total = time.time()

        response_dict=dict(status="ok", response_time_total=str(timedelta(seconds=end_total - start_total)),
                           phase_name=first_phase['Phase_name'],correlation=first_phase['correlation'])

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


