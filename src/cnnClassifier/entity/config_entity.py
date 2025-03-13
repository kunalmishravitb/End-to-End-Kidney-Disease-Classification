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


