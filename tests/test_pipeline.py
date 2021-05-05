from pathlib import Path, PosixPath
from pytest_mock import MockerFixture
import numpy as np
import pandas as pd
import pendulum
from whatsappnalysis.pipeline import run, PipelineConfig


def test_run_pipeline(tmp_path: Path, mocker: MockerFixture):
    """ Test running pipeline"""
    # Arrange
    # - fake chat
    test_chat_txt = (
        "2/5/20, 8:38 PM - Author 1: Hello world\n"
        "2/5/20, 8:39 PM - Author 1: I like balloons\n"
        "2/5/20, 8:39 PM - Author 2: I like balloons too!\n"
        "2/5/20, 8:42 PM - Author 3: <Media omitted>\n"
        "2/5/20, 8:42 PM - Author 3: Balloons are terrible\n"
        "2/5/20, 8:45 PM - Author 2: False\n"
    )

    # - pipeline setup
    config = PipelineConfig(
        root_dir=tmp_path,
        chat_name="test_chat",
        input_chat_text_dir=tmp_path / "raw",
        run_loader=True,
        loaded_chat_parquet_dir=tmp_path / "loaded",
        run_features=True,
        features_chat_parquet_dir=tmp_path / "featured",
        run_model_training=True,
        trained_model_pickle_dir=tmp_path / "trained",
        run_model_prediction=True
    )

    # - save chat to text file
    txt_path = config.input_chat_text_path
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w") as file:
        file.write(test_chat_txt)

    # - mock training
    MockLSTMModel = mocker.patch('whatsappnalysis.pipeline.LSTMModel')
    
    # Act
    run(config=config)

    # Assert - check that all output files were created
    assert config.loaded_chat_parquet_path.exists()
    assert config.features_chat_parquet_path.exists()
    assert MockLSTMModel.return_value.train.call_count == 1
    assert MockLSTMModel.return_value.save.call_count == 1


def test_run_pipeline_off(tmp_path: Path, mocker: MockerFixture):
    """ Test running pipeline - turning off nodes"""
    # Arrange

    # - pipeline setup
    config = PipelineConfig(
        root_dir=tmp_path,
        chat_name="test_chat",
        input_chat_text_dir=tmp_path / "raw",
        run_loader=False,
        loaded_chat_parquet_dir=tmp_path / "loaded",
        run_features=False,
        features_chat_parquet_dir=tmp_path / "featured",
        run_model_training=False,
        trained_model_pickle_dir=tmp_path / "trained",
        run_model_prediction=False
    )

    # - mock training
    MockLSTMModel = mocker.patch('whatsappnalysis.pipeline.LSTMModel')
    MockLSTMModel = mocker.patch('whatsappnalysis.pipeline.ChatDataset')
    
    # Act
    run(config=config)

    # Assert - check that all output files were created



def test_pipeline_config():
    """Test pipeline config"""
    # Arrange

    # Act
    config = PipelineConfig(
        root_dir=Path("foo"),
        chat_name="test_chat",
        input_chat_text_dir="raw",
        run_loader=True,
        loaded_chat_parquet_dir="loaded",
        run_features=True,
        features_chat_parquet_dir="featured",
        run_model_training=True,
        trained_model_pickle_dir="trained",
        run_model_prediction=True
    )

    # Assert
    assert type(config.input_chat_text_path) is PosixPath
    assert type(config.loaded_chat_parquet_path) is PosixPath
    assert type(config.features_chat_parquet_path) is PosixPath
    assert type(config.trained_model_pickle_path) is PosixPath