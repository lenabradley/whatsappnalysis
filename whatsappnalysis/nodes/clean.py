from enum import Enum, auto

import numpy as np
import pandas as pd
from loguru import logger
from whatsappnalysis.dataset import ChatDataset
from whatsappnalysis.schema import Schema


class _Column(Enum):
    CHAT_NAME = auto()
    TIMESTAMP = auto()
    AUTHOR = auto()
    MESSAGE = auto()
    HAS_MEDIA = auto()


_column_to_dtype = {
    _Column.CHAT_NAME.name: str,
    _Column.TIMESTAMP.name: np.dtype("datetime64[ns]"),
    _Column.AUTHOR.name: str,
    _Column.MESSAGE.name: str,
    _Column.HAS_MEDIA.name: bool,
}

schema = Schema(_Column, _column_to_dtype)


_media_string = "<Media omitted>"


def run(input_dataset: ChatDataset) -> ChatDataset:
    """ Clean dataset

    Args:
        input_dataset: input chat dataset

    Returns:
        cleaned chat dataset
    """
    logger.info(f"Running node: Load data")

    cleaned_data = _clean_dataset(input_dataset.data)
    output_dataset = ChatDataset(schema=schema).load_from_pandas(cleaned_data)
    return output_dataset


def _clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """ Clean the given dataset

    Args:
        data: input dataset

    Returns:
        cleaned dataset
    """
    # Add column about media content
    data[_Column.HAS_MEDIA.name] = data[_Column.MESSAGE.name].str.contains(
        _media_string
    )
    data[_Column.MESSAGE.name] = data[_Column.MESSAGE.name].str.replace(
        _media_string, ""
    )

    return data
