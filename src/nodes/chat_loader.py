from typing import Optional
from pathlib import Path
import re
from enum import Enum, auto

import numpy as np
from loguru import logger
import pendulum
import pandas as pd


class MessageColumn(Enum):
    CHAT_NAME = auto()
    TIMESTAMP = auto()
    AUTHOR = auto()
    MESSAGE = auto()


MESSAGE_COLUMN_TO_DTYPE = {
    MessageColumn.CHAT_NAME.name: str,
    MessageColumn.TIMESTAMP.name: np.dtype("datetime64[ns]"),
    MessageColumn.AUTHOR.name: str,
    MessageColumn.MESSAGE.name: str
}


class ChatLoader:
    """ Loads / saves data from a WhatsApp chat history.

    Example:
    ::
        >>> from pathlib import Path
        >>> ChatLoader().run(input_path=Path("path/to/chat/history.txt"), output_path=Path("path/to/output/data.parquet"))
    """

    # Regex patterns for matching WhatsApp chat message history text files
    _re_two_digits = r"\d{1,2}"
    _re_date = f"{_re_two_digits}\/{_re_two_digits}\/{_re_two_digits}"
    _re_time = fr"{_re_two_digits}:{_re_two_digits} [AP]M"
    _re_author = r"[\w ]+"
    _re_message_base = r"[\s\S]*?"
    _re_message_lookahead = fr"\n{_re_date}, {_re_time} - "
    _re_message = fr"{_re_message_base}(?={_re_message_lookahead})"
    _re_chat_pattern = fr"\n({_re_date}, {_re_time}) - ({_re_author}): ({_re_message})"
    _datetime_format_string = "M/D/YY, h:m A"

    def __init__(self):
        """WhatsAppChat data to load / save chat data

        """
        self._data: Optional[pd.DataFrame] = None

    def run(self, input_file: Path, output_file: Path) -> None:
        """ Run the node

        Args:
            input_file: input text file path
            output_file: output parquet file path
        """
        logger.info(f"Running {self.__class__} node")
        self._load_from_txt(input_file)
        self._save_to_parquet(output_file)

    def _load_from_txt(self, filepath: Path) -> Optional[pd.DataFrame]:
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
        if matches is None:
            logger.error(f"No messages found in file {filepath}.")
            return

        for timestamp_str, author_str, message_str in matches:

            # Parse datetime
            timestamp = pendulum.from_format(timestamp_str, self._datetime_format_string)

            # Record data dict
            chat_data_dicts.append({
                MessageColumn.CHAT_NAME.name: chat_name,
                MessageColumn.TIMESTAMP.name: timestamp,
                MessageColumn.AUTHOR.name: author_str,
                MessageColumn.MESSAGE.name: message_str
            })

        logger.info(f"Parsed {len(chat_data_dicts)} messages.")

        logger.info(f"Converting to pandas.")

        # Convert to pandas DF
        self._data = pd.DataFrame.from_records(
            chat_data_dicts,
            columns=[col.name for col in MessageColumn],
        )

        # Enforce dtypes
        self._data = self._data.astype(MESSAGE_COLUMN_TO_DTYPE)

        return self._data

    def _save_to_parquet(self, filepath: Path) -> None:
        """Save dataset to parquet

        Args:
            filepath: output filepath to save parquet data
        """
        if self._data is None:
            logger.warning(f"Data is empty, skipping saving to {filepath}")
            return

        with filepath.open('wb') as file:
            self._data.to_parquet(filepath)
        logger.info(f"Saved data to {filepath}")