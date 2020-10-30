from dataclasses import dataclass
from pathlib import Path
import yaml


DEFAULT_PIPELINE_CONFIG_YAML = Path(__file__).parent / "config_pipeline.yaml"
DEFAULT_MODEL_CONFIG_YAML = Path(__file__).parent / "config_model.yaml"


@dataclass
class PipelineConfig:
    """Pipeline configuration setup"""

    input_chat_text_path: Path

    run_loader: bool
    loaded_chat_parquet_path: Path

    run_cleaner: bool
    cleaned_chat_parquet_path: Path

    run_features: bool
    features_chat_parquet_path: Path

    run_model_setup: bool
    model_input_path: Path

    run_model_training: bool
    trained_model_path: Path

    run_model_prediction: bool

    @classmethod
    def from_yaml(cls, path: Path):
        """Create configuration from yaml path"""
        with path.open() as config_file:
            yaml_contents = yaml.safe_load(config_file)

        config_params = {
            field.name: field.type(yaml_contents.get(field.name))
            for _, field in cls.__dataclass_fields__.items()  # type: ignore
        }
        return cls(**config_params)


@dataclass
class ModelParameters:
    """Hold LSTM model parameters/settings

    Args:
        sequence_length: length of training sequence history
        num_layers: number of LSTM layers, >=1
        num_units: number of units per layer
        dropout_fraction: fractional dropout rate for dropout layers
        activation: string name of activation to use
        loss: string name of loss to use
        optimizer: str name of optimizer to use
        epochs: int, number of epochs for training
    """
    sequence_length: int
    num_layers: int
    num_units: int
    dropout_fraction: float
    activation: str
    loss: str
    optimizer: str
    epochs: int

    @classmethod
    def from_yaml(cls, path: Path):
        """Create configuration from yaml path"""
        with path.open() as config_file:
            yaml_contents = yaml.safe_load(config_file)

        config_params = {
            field.name: field.type(yaml_contents.get(field.name))
            for _, field in cls.__dataclass_fields__.items()  # type: ignore
        }
        return cls(**config_params)



PIPELINE_CONFIG = PipelineConfig.from_yaml(DEFAULT_PIPELINE_CONFIG_YAML)
MODEL_PARAMETERS = ModelParameters.from_yaml(DEFAULT_MODEL_CONFIG_YAML)
