import abc
from pathlib import Path


@dataclass
class BaseConfig:
    """ Base configuration """

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
