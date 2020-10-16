from pathlib import Path

import numpy as np
import pandas as pd
import pendulum
from whatsappnalysis.config import PipelineConfig
from whatsappnalysis.pipeline import run


def test_run_pipeline(tmp_path: Path):
    """ Test running pipeline - integration test """
    # Arrange
    test_chat_txt = (
        "2/5/20, 8:38 PM - Author 1: Hello world\n"
        "2/5/20, 8:39 PM - Author 1: I like balloons\n"
        "2/5/20, 8:39 PM - Author 2: I like balloons too!\n"
        "2/5/20, 8:42 PM - Author 3: <Media omitted>\n"
        "2/5/20, 8:42 PM - Author 3: Balloons are terrible\n"
        "2/5/20, 8:45 PM - Author 2: False\n"
    )
    txt_path = tmp_path / "01_raw" / "test_chat.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w") as file:
        file.write(test_chat_txt)
    config = PipelineConfig(
        input_chat_text_path=txt_path,
        run_loader=True,
        loaded_chat_parquet_path=tmp_path / "loaded" / "test_chat.parquet",
        run_cleaner=True,
        cleaned_chat_parquet_path=tmp_path / "cleaned" / "test_chat.parquet",
        run_features=True,
        features_chat_parquet_path=tmp_path / "featured" / "test_chat.parquet",
    )

    # Act
    run(config=config)

    # Assert - check that all output files were created
    assert config.loaded_chat_parquet_path.exists()
    assert config.cleaned_chat_parquet_path.exists()
    assert config.features_chat_parquet_path.exists()


def test_run_pipeline_load(tmp_path: Path):
    """ Test running pipeline, loading pre-saved files - integration test """
    # Arrange
    test_chat_data = pd.DataFrame.from_dict(
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
                0: pendulum.parse("2020-02-05 20:38:00"),
                1: pendulum.parse("2020-02-05 20:39:00"),
                2: pendulum.parse("2020-02-05 20:39:00"),
                3: pendulum.parse("2020-02-05 20:42:00"),
                4: pendulum.parse("2020-02-05 20:42:00"),
                5: pendulum.parse("2020-02-05 20:45:00"),
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
                3: "",
                4: "Balloons are terrible",
                5: "False",
            },
            "HAS_MEDIA": {0: False, 1: False, 2: False, 3: True, 4: False, 5: False},
            "OVERALL_POLARITY": {
                0: 0.0,
                1: 0.3612,
                2: 0.4199,
                3: 0.0,
                4: -0.4767,
                5: 0.0,
            },
            "NEGATIVE_POLARITY": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.608, 5: 0.0},
            "NEUTRAL_POLARITY": {0: 1.0, 1: 0.286, 2: 0.417, 3: 0.0, 4: 0.392, 5: 1.0},
            "POSITIVE_POLARITY": {0: 0.0, 1: 0.714, 2: 0.583, 3: 0.0, 4: 0.0, 5: 0.0},
            "TOKENS": {
                0: ["Hello world"],
                1: ["I like balloons"],
                2: ["I like balloons too!"],
                3: [""],
                4: ["Balloons are terrible"],
                5: ["False"],
            },
            "TOKEN_POLARITIES": {
                1: [0.3612],
                2: [0.4199],
                3: [],
                4: [-0.4767],
                5: [0.0],
            },
            "WORD_COUNT": {0: 2, 1: 3, 2: 4, 3: 1, 4: 3, 5: 1},
        }
    ).astype({"TIMESTAMP": np.dtype("datetime64[ns]")})

    config = PipelineConfig(
        input_chat_text_path=tmp_path / "01_raw" / "test_chat.txt",
        run_loader=False,
        loaded_chat_parquet_path=tmp_path / "loaded" / "test_chat.parquet",
        run_cleaner=False,
        cleaned_chat_parquet_path=tmp_path / "cleaned" / "test_chat.parquet",
        run_features=False,
        features_chat_parquet_path=tmp_path / "featured" / "test_chat.parquet",
    )

    paths = [
        config.loaded_chat_parquet_path,
        config.cleaned_chat_parquet_path,
        config.features_chat_parquet_path,
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            test_chat_data.to_parquet(file)

    # Act
    run(config=config)

    # Assert - check that all output files were created
    assert config.loaded_chat_parquet_path.exists()
    assert config.cleaned_chat_parquet_path.exists()
    assert config.features_chat_parquet_path.exists()
