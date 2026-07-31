import json
import os
from pathlib import path
from base_loader import BaseLoader

class jsonloader(BaseLoader):
    def load(path):
        document= json.load()