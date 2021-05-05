from pathlib import Path
from enum import Enum, auto
from typing import Dict, Optional
from dataclasses import dataclass
import json
import warnings
import string

import numpy as np
import pandas as pd
from loguru import logger
from keras.utils import np_utils
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint

from whatsappnalysis.lib.custom_types import ChatDataset, Schema, Config, Model

# ==== Model configuration
@dataclass
class LSTMModelConfig(Config):
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
    fraction_to_train_on: float


DEFAULT_MODEL_CONFIG_YAML = Path(__file__).parent / "lstm_config.yaml"
DEFAULT_LSTM_CONFIG = LSTMModelConfig.from_yaml(DEFAULT_MODEL_CONFIG_YAML)

# ==== Input data schema
class _Column(Enum):
    TIMESTAMP = auto()
    MESSAGE = auto()


_column_to_dtype = {
    _Column.TIMESTAMP.name: np.dtype("datetime64[ns]"),
    _Column.MESSAGE.name: str,
}

schema = Schema(_Column, _column_to_dtype)


# ==== Model training input format
@dataclass
class ModelInputData:
    """Hold input data for model training

    Args:
        train: (N x sequence_length) array of character integers for training
        target: (N x num_characters) one-hot-encoded array
    """
    train: np.ndarray
    target: np.ndarray


# ==== Model
class LSTMModel(Model):
    """" LSTM Model for text generation """
    predict_seed: str = "Pokemon nobody likes evolves into pants nobody likes"
    model_filename = 'model.h5'
    char_map_filename = 'char_map.json'

    def __init__(self, save_directory: Path, config: LSTMModelConfig = DEFAULT_LSTM_CONFIG) -> None:
        """ Initialize 
        
        Args:
            path: Path to save/load model from
            config: model configuration parameters
        """
        self.model: Optional[Sequential] = None
        self.input_dataset: Optional[ChatDataset] = None
        self.model_input: Optional[ModelInputData] = None
        self.full_text: Optional[str] = None
        self.config = config
        self.save_directory = save_directory
        self.char_map = self._create_char_map()
        self.checkpoint: Optional[ModelCheckpoint] = None

    def train(self, data: ChatDataset) -> Sequential:
        """ Create and train LSTM model

        Args:
            data: input chat data

        Returns:
            trained model
        """
        self.input_dataset = data

        logger.info(f"Setting up LSTM model input")
        self._setup_input()

        logger.info(f"Creating LSTM model")
        self._create_model()

        logger.info(f"Training LSTM model")
        self._train_model()

        return self.model

    def predict(self, length: Optional[int] = None, seed: Optional[str] = None) -> str:
        """ Generate text from the given model

        Returns:
            generated text
        """
        logger.info("Running node: Generating text")
        generated_text = self._generate_text(
            predict_length=length if length is not None else self.config.sequence_length,
            seed=seed if seed is not None else self.predict_seed
        )
        return generated_text

    def save(self, path: Optional[Path] = None) -> None:
        """Save model and character map"""
        save_dir = path if path is not None else self.save_directory

        # Create output directory
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save(str(save_dir / self.model_filename))
        else:
            warnings.warn(f"Model not yet created/trained, skipping save.")

        char_path = save_dir / self.char_map_filename
        with char_path.open('w') as file:
            json.dump(self.char_map, file)
        
        logger.info(f"Saved model and char map to: {save_dir}")

    def load(self, path: Optional[Path] = None) -> None:
        """Load model and char array from file"""
        save_dir = path if path is not None else self.save_directory

        self.model = load_model(str(save_dir / self.model_filename))

        char_path = save_dir / self.char_map_filename
        with char_path.open() as file:
            char_map = json.load(file)
        self.char_map = char_map
        
        logger.info(f"Loaded model and char map from: {save_dir}")

    def _setup_input(self) -> None:
        """ Create model input  """
        logger.info("Collecting full text")
        self._get_full_text()

        logger.info("Creating model input")
        self._create_model_input()

    def _get_full_text(self) -> None:
        """ Combine all test into one string """
        all_messages = ""
        data = self.input_dataset.data
        sorted_data = data.sort_values(by=_Column.TIMESTAMP.name)
        for _, row in sorted_data.iterrows():
            message_lower = row[_Column.MESSAGE.name].lower()
            all_messages = " ".join((all_messages, message_lower))

        # replace all whitespace with a single space
        all_messages = " ".join(all_messages.split())

        self.full_text = all_messages

    def _create_char_map(self) -> None:
        """Create character / int mapping """
        # Character map of all pritnable characters
        return {char: n for n, char in enumerate(string.printable)}
    
    def _create_model_input(self) -> None:
        """ Create model input arrays and character mapping """
        char_map = self.char_map
        text = self.full_text
        parameters = self.config

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

        self.model_input = ModelInputData(train, target)

    def _create_and_train_model(self) -> None:
        """

        Args:
            data: model data
            parameters: model parameters
            save_path: path to save intermediary model training

        Returns:
            sequential model object
        """

    def _create_model(self) -> None:
        """ Create model layers """
        parameters = self.config
        data = self.model_input
        save_path = self.save_directory
        
        logger.info(f"Creating model with parameters {parameters}.")

        # Setup model
        self.model = Sequential()
        model = self.model

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

        # Checkpoint for saving during training
        logger.info(f"Intermediary models will be saved to {save_path}.")
        self.checkpoint = ModelCheckpoint(
            str(save_path),
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='auto',
            save_freq='epoch'
        )

        # Add final layers
        model.add(Dense(data.target.shape[1], activation=parameters.activation))
        model.compile(loss=parameters.loss, optimizer=parameters.optimizer)
    
    def _train_model(self) -> None:  # pragma: no cover
        """Train model"""
        data = self.model_input
        model = self.model
        parameters = self.config

        keep = int(self.config.fraction_to_train_on * len(data.train))
        logger.info(f"Training model on {int(self.config.fraction_to_train_on * 100)}% of the data.")
        model.fit(
            self._normalize(data.train[:keep]),
            data.target[:keep],
            epochs=parameters.epochs,
            batch_size=parameters.sequence_length,
            callbacks=[self.checkpoint]
        )

    def _normalize(self, array: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Normalize the given data array for model training/prediction"""
        return array / len(self.char_map)

    def _generate_text(self, predict_length: int, seed: str) -> str:
        """ Use the model to generate text

        Returns:
            str of generated text
        """
        character_map = self.char_map
        model = self.model

        # Fix up the seed, make sure its the right length
        if len(seed) > self.config.sequence_length:
            # if seed is too long, just take the end of it
            seed = seed[-self.config.sequence_length:]
        if len(seed) < self.config.sequence_length:
            # if seed is too short, pad it with spaces
            seed = seed.rjust(self.config.sequence_length, " ")
        assert len(seed) == self.config.sequence_length
        logger.info(f"Text generation seed: `{seed}`")


        # Parameters
        num_to_char_map = {num: char for char, num in character_map.items()}

        # Setup seed
        seed_numbers = [character_map[char] for char in seed.lower()]

        # Generate characters
        generated_sequence = ''.join([num_to_char_map[value] for value in seed_numbers])
        for _ in range(predict_length):

            # Pick most likely next character (encoded as a number)
            x = np.reshape(seed_numbers, (1, len(seed_numbers), 1))
            x = self._normalize(x)
            pred_index = np.argmax(model.predict(x, verbose=0))

            # Add predicted next character
            generated_sequence += num_to_char_map[pred_index]

            # Update seed
            seed_numbers.pop(0)
            seed_numbers.append(pred_index)

        generated_string = ''.join(generated_sequence)
        return generated_string
