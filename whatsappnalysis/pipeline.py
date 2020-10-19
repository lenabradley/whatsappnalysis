import pickle
from loguru import logger
import keras
from whatsappnalysis.config import PIPELINE_CONFIG, PipelineConfig
from whatsappnalysis.dataset import ChatDataset
from whatsappnalysis.nodes import add_features, clean, load, model_lstm


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
        with config.model_input_path.open('wb') as file:
            pickle.dump(model_input, file)
    else:
        with config.model_input_path.open('rb') as file:
            model_input = pickle.load(file)

    if config.run_model_training:
        model = model_lstm.train(model_input=model_input)
        model.save(config.trained_model_path)
    else:
        model = keras.models.load_model(config.trained_model_path)

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
