from pathlib import Path
from enum import Enum, auto

from loguru import logger
import numpy as np

from src.dataset import ChatDataset
from src.schema import Schema


class _Column(Enum):
    CHAT_NAME = auto()
    TIMESTAMP = auto()
    AUTHOR = auto()
    MESSAGE = auto()


_column_to_dtype = {
    _Column.CHAT_NAME.name: str,
    _Column.TIMESTAMP.name: np.dtype("datetime64[ns]"),
    _Column.AUTHOR.name: str,
    _Column.MESSAGE.name: str
}


schema = Schema(_Column, _column_to_dtype)


def run(input_file: Path) -> ChatDataset:
    """ Load chat data from text file

    Args:
        input_file: input text file path

    Returns:
        loaded chat dataset
    """
    logger.info(f"Running node: Load data")
    dataset = ChatDataset(schema=schema)
    dataset.load_from_txt(input_file)
    return dataset

