from loguru import logger

from src.config import PIPELINE_CONFIG as config
from src.dataset import ChatDataset
from src.nodes import load, clean, add_features


def run():

    if config.run_loader:
        dataset = load.run(input_file=config.input_chat_text_path)
        dataset.save_to_parquet(config.loaded_chat_parquet_path)
    else:
        dataset = ChatDataset(schema=load.schema).load_from_parquet(config.loaded_chat_parquet_path)

    if config.run_cleaner:
        dataset = clean.run(input_dataset=dataset)
        dataset.save_to_parquet(config.cleaned_chat_parquet_path)
    else:
        dataset = ChatDataset(schema=clean.schema).load_from_parquet(config.cleaned_chat_parquet_path)

    if config.run_features:
        dataset = add_features.run(input_dataset=dataset)
        dataset.save_to_parquet(config.features_chat_parquet_path)
    else:
        dataset = ChatDataset(schema=add_features.schema).load_from_parquet(config.features_chat_parquet_path)


    logger.info(f"Pipeline complete.")


if __name__ == '__main__':
    run()
