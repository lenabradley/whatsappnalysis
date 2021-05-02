
from dataclasses import dataclass
from pathlib import Path
import pickle
from loguru import logger
import keras
from whatsappnalysis.config import PIPELINE_CONFIG, PipelineConfig
from whatsappnalysis.dataset import ChatDataset
from whatsappnalysis.nodes import add_features, clean, load, model_lstm

from lib.base_config import BaseConfig


@dataclass
class PipelineConfig(BaseConfig):
    """Pipeline configuration setup"""

    root_dir: Path
    chat_name: str

    input_chat_text_dir: Path

    run_loader: bool
    loaded_chat_parquet_dir: Path

    run_cleaner: bool
    cleaned_chat_parquet_dir: Path

    run_features: bool
    features_chat_parquet_dir: Path

    run_model_setup: bool
    model_input_pickle_dir: Path

    run_model_training: bool
    trained_model_pickle_dir: Path

    run_model_prediction: bool

    @property
    def input_chat_text_path(self):
        return self.root_dir / self.input_chat_text_dir / (self.chat_name + ".txt")

    @property
    def loaded_chat_parquet_path(self):
        return self.root_dir / self.loaded_chat_parquet_dir / (self.chat_name + ".parquet")

    @property
    def cleaned_chat_parquet_path(self):
        return self.root_dir / self.cleaned_chat_parquet_dir / (self.chat_name + ".parquet")

    @property
    def features_chat_parquet_path(self):
        return self.root_dir / self.features_chat_parquet_dir / (self.chat_name + ".parquet")

    @property
    def model_input_pickle_path(self):
        return self.root_dir / self.model_input_pickle_dir / (self.chat_name + "_model_input.pkl")

    @property
    def trained_model_pickle_path(self):
        return self.root_dir / self.trained_model_pickle_dir / (self.chat_name + "_model.pkl")

DEFAULT_PIPELINE_CONFIG_YAML = Path(__file__).parent / "config_pipeline.yaml"
PIPELINE_CONFIG = PipelineConfig.from_yaml(DEFAULT_PIPELINE_CONFIG_YAML)


@logger.catch
def run(config: PipelineConfig = PIPELINE_CONFIG) -> None:

    if config.run_loader:
        dataset = load.run(input_file=config.input_chat_text_path)
        dataset.save_to_parquet(config.loaded_chat_parquet_path)
    else:
        dataset = ChatDataset(schema=load.schema).load_from_parquet(
            config.loaded_chat_parquet_path
        )

    if config.run_cleaner:
        dataset = clean.run(input_dataset=dataset)
        dataset.save_to_parquet(config.cleaned_chat_parquet_path)
    else:
        dataset = ChatDataset(schema=clean.schema).load_from_parquet(
            config.cleaned_chat_parquet_path
        )

    if config.run_features:
        dataset = add_features.run(input_dataset=dataset)
        dataset.save_to_parquet(config.features_chat_parquet_path)
    else:
        dataset = ChatDataset(schema=add_features.schema).load_from_parquet(
            config.features_chat_parquet_path
        )

    if config.run_model_setup:
        model_input = model_lstm.setup_input(input_dataset=dataset)
        config.model_input_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with config.model_input_pickle_path.open('wb') as file:
            pickle.dump(model_input, file)
    else:
        with config.model_input_pickle_path.open('rb') as file:
            model_input = pickle.load(file)

    if config.run_model_training:
        model = model_lstm.train(
            model_input=model_input,
            save_path=config.trained_model_pickle_path
        )
        model.save(config.trained_model_pickle_path)
    else:
        model = keras.models.load_model(config.trained_model_pickle_path)

    if config.run_model_prediction:
        text = model_lstm.predict(
            model=model,
            input_data=model_input,
            seed="pie for breakfast is "
        )
        logger.info(f"Generated text:\n{text}")

    logger.info(f"Pipeline complete.")


if __name__ == "__main__":
    run()
