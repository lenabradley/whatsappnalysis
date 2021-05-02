import abc
import typing as t
import pathlib

class BaseModel(ABC):
    
    @abc.abstractmethod
    def train(self) -> t.Any:

    @abc.abstractmethod
    def predict(self) -> str:

    @abc.abstractmethod
    def save(self, path: pathlib.Path) -> None:

    @abc.abstractmethod
    def load(self, path: pathlib.Path) -> None:

        
