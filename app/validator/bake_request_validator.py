import logging
from typing import Optional
from pydantic import BaseModel, validator

# getting logger configuration
logger = logging.getLogger(__name__)

class PayloadAnomalyRequest(BaseModel):
    train_set: Optional[str]
    test_set: Optional[str]
    single_measurement : Optional[dict]

    @validator('train_set', pre=True)
    def train_set_val(cls, v):
        if type(v) is not str:
            raise TypeError("train_set must be str, not " + type(v).__name__)
        return v

    @validator('test_set', pre=True)
    def test_set_val(cls, v):
        if type(v) is not str:
            raise TypeError("test_set must be str, not " + type(v).__name__)
        return v

    @validator('single_measurement', pre=True)
    def single_measurement_val(cls, v):
        if type(v) is not dict:
            raise TypeError("single_measurement must be str, not " + type(v).__name__)
        return v


class PayloadStatsRequest(BaseModel):
    file_name: str
    daily: Optional[bool]=True
    test_size : Optional[float]=0.5
    outliers: Optional[bool]=False


    @validator('file_name', pre=True)
    def file_name_val(cls, v):
        if type(v) is not str:
            raise TypeError("file_name must be str, not " + type(v).__name__)
        return v

    @validator('test_size', pre=True)
    def test_size_val(cls, v):
        if type(v) is not float:
            raise TypeError("test_size must be float, not " + type(v).__name__)
        return v

    @validator('daily', pre=True)
    def daily_val(cls, v):
        if type(v) is not bool:
            raise TypeError("daily must be bool, not " + type(v).__name__)
        return v

    @validator('outliers', pre=True)
    def outliers_val(cls, v):
        if type(v) is not bool:
            raise TypeError("outliers must be bool, not " + type(v).__name__)
        return v

