from enum import Enum, auto
from typing import Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from whatsappnalysis.dataset import ChatDataset
from whatsappnalysis.schema import Schema


class _Column(Enum):
    TIMESTAMP = auto()
    MESSAGE = auto()


_column_to_dtype = {
    _Column.TIMESTAMP.name: np.dtype("datetime64[ns]"),
    _Column.MESSAGE.name: str,
}

schema = Schema(_Column, _column_to_dtype)


@dataclass
class ModelInputData:
    """Hold input data for model training

    Args:
        train: (N x sequence_length) array of character integers for training
        target: (N x num_characters) one-hot-encoded array
        character_map: Dictionary mapping characters to integers
    """
    train: np.ndarray
    target: np.ndarray
    character_map: Dict[str, int]


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
    sequence_length: int = 50
    num_layers: int = 3
    num_units: int = 700
    dropout_fraction: float = 0.2
    activation: str = 'softmax'
    loss: str = 'categorical_crossentropy'
    optimizer: str = 'adam'
    epochs: int = 30


def setup_input(input_dataset: ChatDataset) -> ModelInputData:
    """ Create model input

    Args:
        input_dataset: input chat dataset

    Returns:
        input model data
    """
    logger.info(f"Running node: Setup model input")
    data = input_dataset.data

    logger.info("Collecting full text")
    full_text = _get_full_text(data)

    logger.info("Creating model input")
    parameters = ModelParameters()
    model_input = _create_model_input(text=full_text, parameters=parameters)
    return model_input


def train(model_input: ModelInputData, parameters: ModelParameters = ModelParameters()) -> Sequential:
    """ Train LSTM model

    Args:
        model_input: input model data

    Returns:
        trained model
    """
    logger.info(f"Running node: Training model")
    model = _create_and_train_model(data=model_input, parameters=parameters)

    return model


def predict(
        input_data: ModelInputData,
        model: Sequential,
        seed: str,
        length: int = 100
) -> str:
    """ Generate text from the given model

    Args:
        input_data: model input data
        model: trained model
        seed: string to seed text generation
        length: length of string to generate


    Returns:
        generated text
    """
    logger.info("Running node: Generating text")
    generated_text = _generate_text(
        data=input_data,
        model=model,
        predict_length=length,
        seed=seed
    )
    return generated_text


def _get_full_text(data: pd.DataFrame) -> str:
    """ Combine all test into one string """
    all_messages = ""
    sorted_data = data.sort_values(by=_Column.TIMESTAMP.name)
    for _, row in sorted_data.iterrows():
        message_lower = row[_Column.MESSAGE.name].lower()
        all_messages = " ".join((all_messages, message_lower))

    # replace all whitespace with a single space
    all_messages = " ".join(all_messages.split())

    return all_messages


def _create_model_input(text: str, parameters: ModelParameters) -> ModelInputData:
    """ Create model input arrays and character mapping

    Args:
        text: string, full text to use for training
        parameters: model parameters

    Returns:
        _ModelInput object based on the input text
    """
    # Character map
    characters = sorted(list(set(text)))
    char_map = {char: n for n, char in enumerate(characters)}

    # Create train/target arrays
    train = []
    target = []
    for index in range(len(text) - parameters.sequence_length):
        seq = text[index:index + parameters.sequence_length]
        label = text[index + parameters.sequence_length]

        train.append(
            [char_map[char] for char in seq]
        )
        target.append(char_map[label])

    train = np.reshape(train, (len(train), parameters.sequence_length, 1))
    target = np_utils.to_categorical(target)

    return ModelInputData(train, target, char_map)


def _create_and_train_model(data: ModelInputData, parameters: ModelParameters) -> Sequential:
    """

    Args:
        data: model data
        parameters: model parameters

    Returns:
        sequential model object
    """
    logger.info("Creating model layers.")
    # Setup model
    model = Sequential()

    # Add initial layer
    model.add(LSTM(
        parameters.num_units,
        input_shape=(data.train.shape[1], data.train.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(parameters.dropout_fraction))

    # Add subsequent layers
    assert parameters.num_units >= 1, \
        f"Minumum of 1 layer required, but parameters specify {parameters.num_layers} layers."
    for index in range(parameters.num_layers - 1):
        return_sequences = index < parameters.num_layers - 2 
        model.add(LSTM(parameters.num_units, return_sequences=return_sequences))
        model.add(Dropout(parameters.dropout_fraction))

    # Add final layers
    model.add(Dense(data.target.shape[1], activation=parameters.activation))
    model.compile(loss=parameters.loss, optimizer=parameters.optimizer)

    # Train model
    logger.info("Training model.")
    model.fit(
        _normalize(data.train, data),
        data.target,
        epochs=parameters.epochs,
        batch_size=parameters.sequence_length
    )

    return model


def _normalize(array: np.ndarray, data: ModelInputData) -> np.ndarray:
    """Normalize the given data array for model training/prediction"""
    return array / len(data.character_map)


def _generate_text(
        data: ModelInputData,
        model: Sequential,
        predict_length: int,
        seed: str
) -> str:
    """ Use the given model to generate text

    Args:
        data: training data
        model: trained model
        predict_length: number of characters to generate
        seed: string to kick-off generation

    Returns:
        str of generated text
    """
    # Parameters
    num_to_char_map = {num: char for char, num in data.character_map.items()}

    # Setup seed
    seed_numbers = [data.character_map[char] for char in seed.lower()]

    # Generate characters
    generated_sequence = ''.join([num_to_char_map[value] for value in seed_numbers])
    for _ in range(predict_length):

        # Pick most likely next character (encoded as a number)
        x = np.reshape(seed_numbers, (1, len(seed_numbers), 1))
        x = _normalize(x, data)
        pred_index = np.argmax(model.predict(x, verbose=0))

        # Add predicted next character
        generated_sequence += num_to_char_map[pred_index]

        # Update seed
        seed_numbers.pop(0)
        seed_numbers.append(pred_index)

    generated_string = ''.join(generated_sequence)
    return generated_string
