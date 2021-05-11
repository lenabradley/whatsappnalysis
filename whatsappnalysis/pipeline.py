from dataclasses import dataclass
from pathlib import Path
import pickle
from loguru import logger
import keras
from .lib.custom_types import Config, ChatDataset
from .lib.analysis import FeatureAdder
from .lib.model import LSTMModel
from .lib.data_loader import WhatsappLoader, whatsapp_schema


@dataclass
class PipelineConfig(Config):
    """Pipeline configuration setup"""

    root_dir: Path
    chat_name: str

    input_chat_text_dir: Path

    run_loader: bool
    loaded_chat_parquet_dir: Path

    run_features: bool
    features_chat_parquet_dir: Path

    run_model_training: bool
    trained_model_pickle_dir: Path

    run_model_prediction: bool
    length: int
    seed: str

    @property
    def input_chat_text_path(self):
        return self.root_dir / self.input_chat_text_dir / (self.chat_name + ".txt")

    @property
    def loaded_chat_parquet_path(self):
        return (
            self.root_dir / self.loaded_chat_parquet_dir / (self.chat_name + ".parquet")
        )

    @property
    def features_chat_parquet_path(self):
        return (
            self.root_dir
            / self.features_chat_parquet_dir
            / (self.chat_name + ".parquet")
        )

    @property
    def trained_model_pickle_path(self):
        return (
            self.root_dir
            / self.trained_model_pickle_dir
            / (self.chat_name + "_model.pkl")
        )


DEFAULT_PIPELINE_CONFIG_YAML = Path(__file__).parent / "pipeline_config.yaml"
PIPELINE_CONFIG = PipelineConfig.from_yaml(DEFAULT_PIPELINE_CONFIG_YAML)


@logger.catch
def run(config: PipelineConfig = PIPELINE_CONFIG) -> None:

    # Load from file
    if config.run_loader:
        dataset = WhatsappLoader().load_from_txt(config.input_chat_text_path)
        dataset.save_to_parquet(config.loaded_chat_parquet_path)
    else:
        dataset = ChatDataset(whatsapp_schema).load_from_parquet(
            config.loaded_chat_parquet_path
        )

    # Add features
    if config.run_features:
        dataset = FeatureAdder().add_features(dataset)
        dataset.save_to_parquet(config.features_chat_parquet_path)

    # Train model
    if config.run_model_training:
        model = LSTMModel(save_directory=config.trained_model_pickle_dir)
        model.train(dataset)
        model.save()

    else:
        model = LSTMModel(save_directory=config.trained_model_pickle_dir)
        model.load()

    # Predict
    if config.run_model_prediction:
        text = model.predict(length=config.length, seed=config.seed)
        logger.info(f"Generated text:\n``{text}``")

    logger.info(f"Pipeline complete.")


if __name__ == "__main__":
    run()
