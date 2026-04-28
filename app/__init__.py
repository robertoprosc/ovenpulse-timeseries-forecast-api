from fastapi import FastAPI
from fastapi_utils.api_settings import get_api_settings

from config import Settings
from app.views import preprocess_and_calculate_statistics_router,anomaly_forecast_router

settings = Settings()

get_api_settings.cache_clear()
app_settings = get_api_settings()
app_settings.debug = True

# Registering routers
app = FastAPI(**app_settings.fastapi_kwargs)


app.include_router(preprocess_and_calculate_statistics_router.router, prefix="/stats", tags=["statistics"])
app.include_router(anomaly_forecast_router.router, prefix="/anomaly", tags=["anomaly-forecast"])

