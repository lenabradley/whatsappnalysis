from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd
import pendulum
import pytest
from whatsappnalysis.lib.custom_types import ChatDataset, Schema


class TestChatDataset:
    """ Tests for ChatDataset """

    test_chat_df = pd.DataFrame.from_dict(
        {
            "CHAT_NAME": {
                0: "test_chat",
                1: "test_chat",
                2: "test_chat",
                3: "test_chat",
                4: "test_chat",
                5: "test_chat",
            },
            "TIMESTAMP": {
                0: pendulum.parse("2020-02-05 20:38:00+0000"),
                1: pendulum.parse("2020-02-05 20:39:00+0000"),
                2: pendulum.parse("2020-02-05 20:39:00+0000"),
                3: pendulum.parse("2020-02-05 20:42:00+0000"),
                4: pendulum.parse("2020-02-05 20:42:00+0000"),
                5: pendulum.parse("2020-02-05 20:45:00+0000"),
            },
            "AUTHOR": {
                0: "Author 1",
                1: "Author 1",
                2: "Author 2",
                3: "Author 3",
                4: "Author 3",
                5: "Author 2",
            },
            "MESSAGE": {
                0: "Hello world",
                1: "I like balloons",
                2: "I like balloons too!",
                3: "foo",
                4: "Balloons are terrible",
                5: "False",
            },
        }
    )

    class Columns(Enum):
        TIMESTAMP = auto()
        AUTHOR = auto()
        MESSAGE = auto()

    schema = Schema(
        columns=Columns,
        columns_to_dtypes={Columns.TIMESTAMP.name: np.dtype("datetime64[ns]")},
    )

    def test_load_from_parquet(self, tmp_path: Path):
        """ Test loading from parquet file"""
        # Arrange
        expected = self.test_chat_df.astype({"TIMESTAMP": np.dtype("datetime64[ns]")})
        raw_path = tmp_path / "test_chat.parquet"
        with raw_path.open("wb") as file:
            expected.to_parquet(file)

        dataset = ChatDataset(schema=self.schema)

        # Act
        result = dataset.load_from_parquet(raw_path)

        # Assert
        pd.testing.assert_frame_equal(result.data, expected)

    def test_load_from_pandas(self):
        """ Test loading from pandas DF"""
        # Arrange
        expected = self.test_chat_df.astype({"TIMESTAMP": np.dtype("datetime64[ns]")})

        dataset = ChatDataset(schema=self.schema)

        # Act
        result = dataset.load_from_pandas(expected)

        # Assert
        pd.testing.assert_frame_equal(result.data, expected)

    def test_bad_schema(self):
        """ Test error with bad schema"""
        # Arrange
        data = self.test_chat_df
        data = data.drop(columns="TIMESTAMP")

        dataset = ChatDataset(schema=self.schema)

        # Act / assert
        with pytest.raises(ValueError):
            dataset.load_from_pandas(data)
