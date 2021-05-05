from enum import Enum, auto
from pathlib import Path
from typing import Optional

import string
import pytest
from pytest_mock import MockerFixture
import pandas as pd
import pendulum
import numpy as np
from whatsappnalysis.lib.model.lstm import LSTMModel, ChatDataset, Sequential, Schema


class TestLSTMModel:
    """Test LSTMModel"""

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
    
    dataset = ChatDataset(schema).load_from_pandas(test_chat_df)

    fake_train_method = lambda _: None

    def test_train(self, tmp_path: Path, mocker: MockerFixture):
        """Test train"""
        # Arrange
        model = LSTMModel(tmp_path / "test_train")
        mocker.patch.object(model, "_train_model", self.fake_train_method)

        # Act
        result = model.train(self.dataset)

        # Assert
        assert type(result) is Sequential

    def test_save(self, tmp_path: Path, mocker: MockerFixture):
        """Test save"""
        # Arrange
        save_dir = tmp_path / "test_save"
        model = LSTMModel(save_dir)
        mocker.patch.object(model, "_train_model", self.fake_train_method)
        model.train(self.dataset)

        # Act
        model.save()

        # Assert - check for expected files
        filenames = list(p.name for p in save_dir.iterdir())
        assert model.model_filename in filenames
        assert model.char_map_filename in filenames


    def test_save_warning(self, tmp_path: Path):
        """Test save throws warning if model doesn't exist yet"""
        # Arrange
        save_dir = tmp_path / "test_save"
        model = LSTMModel(save_dir) # don't train model

        # Act / assert
        with pytest.warns(UserWarning):
            model.save()

    def test_load(self, tmp_path: Path, mocker: MockerFixture):
        """Test load"""
        # Arrange
        save_dir = tmp_path / "test_load"
        model = LSTMModel(save_dir)
        mocker.patch.object(model, "_train_model", self.fake_train_method)
        model.train(self.dataset)
        model.save()
        new_model = LSTMModel(save_dir)

        # Act
        new_model.load()

        # Assert
        assert new_model.model is not None
        assert new_model.char_map is not None

    @pytest.mark.parametrize('seed', [None, string.printable, "a"])
    def test_predict(self, seed: Optional[str], tmp_path: Path, mocker: MockerFixture):
        """Test predict"""
        # Arrange
        save_dir = tmp_path / "test_pred"
        model = LSTMModel(save_dir)
        mocker.patch.object(model, "_train_model", self.fake_train_method)
        model.train(self.dataset)

        # Act
        result = model.predict(seed=seed)

        # Assert
        assert result
        assert type(result) is str
