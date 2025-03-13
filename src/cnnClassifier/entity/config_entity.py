# entity: return type of the function. So here it is return type of DataIngestionConfig function
from dataclasses import dataclass # dataclass is a decorator. It is a class decorator which helps to define the class with attributes and methods.
from pathlib import Path

# Whatever I have written in the 'config.yaml' file, I am returning it as a class object.
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path




# frozen=True: make the class immutable
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int






@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list






