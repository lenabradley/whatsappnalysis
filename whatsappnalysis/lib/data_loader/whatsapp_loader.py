from enum import Enum, auto
import re
from pathlib import Path

import pandas as pd
from loguru import logger
import pendulum
import numpy as np

from whatsappnalysis.lib.custom_types import ChatDataset, Schema


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
    _Column.HAS_MEDIA.name: bool
}

schema = Schema(_Column, _column_to_dtype)


class WhatsappLoader:
    """Class to load WhatsApp dataset"""


    # Regex patterns for loading WhatsApp chat message history text files
    _re_two_digits = r"\d{1,2}"
    _re_date = fr"{_re_two_digits}\/{_re_two_digits}\/{_re_two_digits}"
    _re_time = fr"{_re_two_digits}:{_re_two_digits} [AP]M"
    _re_author = r"[^:\n]+"
    _re_message_base = r"[\s\S]*?"
    _re_message_lookahead = fr"\n{_re_date}, {_re_time} - "
    _re_message = fr"{_re_message_base}(?={_re_message_lookahead}|$)"
    _re_chat_pattern = fr"\n?({_re_date}, {_re_time}) - ({_re_author}): ({_re_message})"
    _datetime_format_string = "M/D/YY, h:m A"

    # Params for cleaning
    _media_string = "<Media omitted>"

    def __init__(self):
        """WhatsAppChat data to load / save chat data"""
        pass

    def load_from_txt(self, filepath: Path) -> ChatDataset:
        """Loads data from the text file

        Args:
            filepath: path to input text file

        Returns:
            dataframe of chat message following the MessageColumn schema
        """
        logger.info(f"Parsing chat data from {filepath}")

        # Read from file
        with filepath.open() as file:
            file_contents = file.read()

        # Take chat name as the base file name
        chat_name = filepath.stem
        logger.info(f"Chat name: {chat_name}")

        # Find messages in text file
        chat_data_dicts = []
        matches = re.findall(self._re_chat_pattern, file_contents)
        if not matches:
            raise TypeError(f"No messages found in file {filepath}.")

        for timestamp_str, author_str, message_str in matches:

            # Parse datetime
            timestamp = pendulum.from_format(
                timestamp_str, self._datetime_format_string
            )

            # Record data dict
            chat_data_dicts.append(
                {
                    "CHAT_NAME": chat_name,
                    "TIMESTAMP": timestamp,
                    "AUTHOR": author_str,
                    "MESSAGE": message_str,
                }
            )

        logger.info(f"Parsed {len(chat_data_dicts)} messages.")

        logger.info(f"Converting to pandas.")

        # Convert to pandas DF
        data = pd.DataFrame.from_records(chat_data_dicts)

        # Clean
        data = self._clean(data)

        # Create output dataset
        dataset = ChatDataset(schema=schema).load_from_pandas(data)

        return dataset

    def _clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Clean the given dataset

        Args:
            data: input dataset

        Returns:
            cleaned dataset
        """
        # Mark media content
        data[_Column.HAS_MEDIA.name] = data[_Column.MESSAGE.name].str.contains(
            self._media_string
        )

        # Remove media content
        data[_Column.MESSAGE.name] = data[_Column.MESSAGE.name].str.replace(
            self._media_string, ""
        )

        # Remove special characters
        data[_Column.MESSAGE.name] = data[_Column.MESSAGE.name].str.encode('ascii', 'ignore').str.decode('ascii')

        return data