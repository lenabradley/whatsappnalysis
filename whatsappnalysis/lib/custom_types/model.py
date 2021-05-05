import abc
import typing as t
import pathlib
from .chat_dataset import ChatDataset

class Model(abc.ABC):
    
    @abc.abstractmethod
    def train(self, dataset: ChatDataset) -> t.Any:
        pass

    @abc.abstractmethod
    def predict(self) -> str:
        pass

    @abc.abstractmethod
    def save(self, path: pathlib.Path) -> None:
        pass

    @abc.abstractmethod
    def load(self, path: pathlib.Path) -> None:
        pass

        
