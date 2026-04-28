import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    This class is used for environment configurations.

    Attributes
    ----------
    """
    RESOURCES_PATH: str = os.path.join(os.getcwd(), "resources")
    RESULTS_PATH: str = os.path.join(RESOURCES_PATH, "results")

    STATS_PATH: str = os.path.join(RESULTS_PATH,"stats","statistics.json")

    TREND_DETECTION_GRAPHS: str = os.path.join(RESULTS_PATH, "trend_detection_graphs")

