from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd
import pendulum
import pytest
from whatsappnalysis.lib.custom_types import ChatDataset, Schema
from whatsappnalysis.lib.data_loader import WhatsappLoader


class TestWhatsappLoader:
    """ Tests for ChatDataset """

    test_chat_txt = (
        "2/5/20, 8:38 PM - Author 1: Hello world\n"
        "2/5/20, 8:39 PM - Author 1: I like balloons\n"
        "2/5/20, 8:39 PM - Author 2: I like balloons too!\n"
        "2/5/20, 8:42 PM - Author 3: foo\n"
        "2/5/20, 8:42 PM - Author 3: Balloons are terrible\n"
        "2/5/20, 8:45 PM - Author 2: False\n"
    )

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
            "HAS_MEDIA": {
                0: False,
                1: False,
                2: False,
                3: False,
                4: False,
                5: False,
            }            
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

    def test_load_from_txt(self, tmp_path: Path):
        """ Test loading from txt file"""
        # Arrange
        expected = self.test_chat_df.astype({"TIMESTAMP": np.dtype("datetime64[ns]")})
        raw_path = tmp_path / "test_chat.txt"
        with raw_path.open("w") as file:
            file.write(self.test_chat_txt)

        dataset = ChatDataset(schema=self.schema)

        # Act
        result = WhatsappLoader().load_from_txt(raw_path)

        # Assert
        pd.testing.assert_frame_equal(result.data, expected)

    def test_load_from_txt_bad_file(self, tmp_path: Path):
        """ Test loading from txt file"""
        # Arrange
        raw_path = tmp_path / "test_chat.txt"
        with raw_path.open("w") as file:
            file.write("")

        # Act / assert
        with pytest.raises(TypeError):
            WhatsappLoader().load_from_txt(raw_path)
