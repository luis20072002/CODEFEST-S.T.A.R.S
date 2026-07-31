import json
import os
from pathlib import Path
from base_loader import BaseLoader
from ..core.document import Document

class JSONLoader(BaseLoader):
    def load(self,path:Path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        text = self.parse()

    