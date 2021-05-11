from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from .schema import Schema


class ChatDataset:
    """ Chat dataset"""

    def __init__(self, schema: Schema):
        """Initialize

        Args:
            schema: Schema defining columns and dtypes
        """
        self.schema = schema
        self.data: Optional[pd.DataFrame] = None

    def load_from_parquet(self, filepath: Path) -> ChatDataset:
        """Load dataset from parquet

        Args:
            filepath: Path

        Returns:
            dataset as a pandas dataframe
        """
        logger.info(f"Loading data from parquet {filepath}")
        with filepath.open("rb") as file:
            data = pd.read_parquet(file)

        # post process
        data = self._process(data)

        return self

    def load_from_pandas(self, data: pd.DataFrame) -> ChatDataset:
        """Set data based on given dataframe

        Args:
            data: pandas dataframe with dataset

        Returns:
            validated dataset
        """
        # post process
        data = self._process(data)

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
        with filepath.open("wb") as file:
            self.data.to_parquet(file)

    def _process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data after loading"""
        # Validate schema
        data = self._validate_schema(data)

        # Clean
        data = self._clean(data)

        # store
        self.data = data

        return self.data

    def _validate_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check that the dataset has the required columns and types

        Args:
            data: dataset to validate

        Returns:
            dataset with required columns of expected types
        """
        logger.info(f"Validating dataset schema.")

        # Check that all required columns exist
        required_columns = {col.name for col in self.schema.columns}  # type: ignore
        existing_columns = set(data.columns)
        missing_columns = required_columns - existing_columns
        if missing_columns:
            raise ValueError(f"Dataset is missing columns {missing_columns}.")

        # Enforce dtypes
        data = data.astype(self.schema.columns_to_dtypes)

        return data

    def _clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset"""
        return data
