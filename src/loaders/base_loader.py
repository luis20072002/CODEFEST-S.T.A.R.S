from abc import ABC, abstractmethod

from core.document import Document

class BaseLoader(ABC):

    @abstractmethod
    def load(self, path: str) -> Document:
        pass