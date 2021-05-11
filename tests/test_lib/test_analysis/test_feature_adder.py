from enum import Enum, auto
import pandas as pd
import numpy as np
import pendulum
from whatsappnalysis.lib.analysis.feature_adder import FeatureAdder, ChatDataset, Schema


class TestFeatureAdder:

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

    def test_add_features(self):
        """Test adding features"""
        # Arrange
        dataset = ChatDataset(self.schema).load_from_pandas(self.test_chat_df)

        # Act
        result = FeatureAdder().add_features(dataset)

        # Assert - check result has more cols than input
        input_cols = set(list(dataset.data.columns))
        output_cols = set(list(result.data.columns))
        assert output_cols - input_cols
