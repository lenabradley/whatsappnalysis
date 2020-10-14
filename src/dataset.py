from __future__ import annotations
from typing import Optional
from pathlib import Path
import re

from loguru import logger
import pendulum
import pandas as pd

from src.schema import Schema


class ChatDataset:
    """ Loads / saves data from a WhatsApp chat history. """

    # Regex patterns for loading WhatsApp chat message history text files
    _re_two_digits = r"\d{1,2}"
    _re_date = f"{_re_two_digits}\/{_re_two_digits}\/{_re_two_digits}"
    _re_time = fr"{_re_two_digits}:{_re_two_digits} [AP]M"
    _re_author = r"[^:\n]+"
    _re_message_base = r"[\s\S]*?"
    _re_message_lookahead = fr"\n{_re_date}, {_re_time} - "
    _re_message = fr"{_re_message_base}(?={_re_message_lookahead})"
    _re_chat_pattern = fr"\n({_re_date}, {_re_time}) - ({_re_author}): ({_re_message})"
    _datetime_format_string = "M/D/YY, h:m A"

    def __init__(self, schema: Schema):
        """WhatsAppChat data to load / save chat data

        Args:
            schema: Schema defining columns and detypes
        """
        self.schema = schema
        self.data: Optional[pd.DataFrame] = None

    def _validate_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Check that the dataset has the required columns and types

        Args:
            data: dataset to validate

        Returns:
            dataset with required columns of expected types
        """
        logger.info(f"Validating dataset schema.")

        # Check that all required columns exist
        required_columns = {col.name for col in self.schema.columns}
        existing_columns = set(data.columns)
        missing_columns = required_columns - existing_columns
        if missing_columns:
            raise ValueError(f"Dataset is missing columns {missing_columns}.")

        # Enforce dtypes
        data = data.astype(self.schema.columns_to_dtypes)

        return data

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
        if matches is None:
            raise TypeError(f"No messages found in file {filepath}.")

        for timestamp_str, author_str, message_str in matches:

            # Parse datetime
            timestamp = pendulum.from_format(timestamp_str, self._datetime_format_string)

            # Record data dict
            chat_data_dicts.append({
                "CHAT_NAME": chat_name,
                "TIMESTAMP": timestamp,
                "AUTHOR": author_str,
                "MESSAGE": message_str
            })

        logger.info(f"Parsed {len(chat_data_dicts)} messages.")

        logger.info(f"Converting to pandas.")

        # Convert to pandas DF
        self.data = pd.DataFrame.from_records(chat_data_dicts)

        # Validate schema
        self.data = self._validate_schema(self.data)

        return self

    def load_from_parquet(self, filepath: Path) -> ChatDataset:
        """ Load dataset from parquet

        Args:
            filepath: Path

        Returns:
            dataset as a pandas dataframe
        """
        logger.info(f"Loading data from parquet {filepath}")
        with filepath.open("rb") as file:
            data = pd.read_parquet(file)

        # Validate schema
        self.data = self._validate_schema(data)

        return self

    def load_from_pandas(self, data: pd.DataFrame) -> ChatDataset:
        """ Set data based on given dataframe

        Args:
            data: pandas dataframe with dataset

        Returns:
            validated dataset
        """
        self.data = data

        # Validate schema
        self.data = self._validate_schema(data)

        return self

    def save_to_parquet(self, filepath: Path) -> None:
        """Save dataset to parquet

        Args:
            data: pandas dataframe with data
            filepath: output filepath to save parquet data
        """
        logger.info(f"Saving data to parquet {filepath}")

        # Create output directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save
        with filepath.open('wb') as file:
            self.data.to_parquet(file)


