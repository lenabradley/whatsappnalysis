from loguru import logger
from whatsappnalysis.config import PIPELINE_CONFIG, PipelineConfig
from whatsappnalysis.dataset import ChatDataset
from whatsappnalysis.nodes import add_features, clean, load


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

    logger.info(f"Pipeline complete.")


if __name__ == "__main__":
    run()
