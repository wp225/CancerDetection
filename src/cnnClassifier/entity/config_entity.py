from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_model_path: Path
    params_image_size: List
    params_include_top: bool
    params_learning_rate: float
    params_classes: int
    params_weight: str

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epoch: int
    params_batch_size: int
    params_is_augmented: bool
    params_image_size: List


@dataclass
class EvaluationConfig:
    path_to_model : Path
    training_data : Path
    all_params : dict
    mlflow_uri : str
    params_image_size : list
    params_batch_size : int
