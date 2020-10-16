from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
CHAT_NAME = "example_chat"


@dataclass
class PipelineConfig:
    """Pipeline configuration setup"""

    input_chat_text_path: Path = ROOT_DIR / "data" / "01_raw" / (CHAT_NAME + ".txt")

    run_loader: bool = True
    loaded_chat_parquet_path: Path = ROOT_DIR / "data" / "02_loaded" / (
        CHAT_NAME + ".parquet"
    )

    run_cleaner: bool = True
    cleaned_chat_parquet_path: Path = ROOT_DIR / "data" / "03_cleaned" / (
        CHAT_NAME + ".parquet"
    )

    run_features: bool = True
    features_chat_parquet_path: Path = ROOT_DIR / "data" / "04_featured" / (
        CHAT_NAME + ".parquet"
    )


PIPELINE_CONFIG = PipelineConfig()
