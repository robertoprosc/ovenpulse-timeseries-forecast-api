import json
from config import Settings

settings = Settings()

def load_statistics():
    with open(settings.STATS_PATH,"r") as f:
        data = json.load(f)

    return data


