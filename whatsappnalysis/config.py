from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
CHAT_NAME = "Fart Noises for Life"


@dataclass
class PipelineConfig:
    """Pipeline configuration setup"""

    input_chat_text_path: Path = ROOT_DIR / "data" / "01_raw" / (CHAT_NAME + ".txt")

    run_loader: bool = False
    loaded_chat_parquet_path: Path = ROOT_DIR / "data" / "02_loaded" / (
        CHAT_NAME + ".parquet"
    )

    run_cleaner: bool = False
    cleaned_chat_parquet_path: Path = ROOT_DIR / "data" / "03_cleaned" / (
        CHAT_NAME + ".parquet"
    )

    run_features: bool = False
    features_chat_parquet_path: Path = ROOT_DIR / "data" / "04_featured" / (
        CHAT_NAME + ".parquet"
    )

    run_model_setup: bool = False
    model_input_path: Path = ROOT_DIR / "data" / "05_model" / (CHAT_NAME + "_model_input.pkl")

    run_model_training: bool = False
    trained_model_path: Path = ROOT_DIR / "data" / "05_model" / (CHAT_NAME + "_model.pkl")

    run_model_prediction: bool = True


PIPELINE_CONFIG = PipelineConfig()
