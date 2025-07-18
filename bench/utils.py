import time
from typing import List

import h5py
import numpy as np


class Timer:
    def __enter__(self):
        """Record the start time when entering the context."""
        self.start_time = time.time()  # Record the start time
        self.end_time = self.start_time
        self.elapsed_time = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record the end time and calculate the elapsed time when exiting the context."""
        self.end_time = time.time()  # Record the end time
        self.elapsed_time = self.end_time - self.start_time  # Calculate the elapsed time


def format_dict(d: dict) -> str:
    return '_'.join(f'{key}-{value}' for key, value in d.items())


def cos_normalize(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """
    Normalization function for cosine space datasets.
    
    Args:
        x (np.ndarray): The vectors to be normalized.
        min_value (float): This value is fully ignored.
        max_value (float): This value is fully ignored.
    
    Returns:
        np.ndarray: Normalized vectors with zero vectors set to a large-enough number (100).
    """
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    zero_mask = (norm == 0)
    norm[zero_mask] = 1
    normalized_x = x / norm
    normalized_x[zero_mask.squeeze()] = np.max(normalized_x[~zero_mask.squeeze()])
    
    return normalized_x


def read_hdf5_dataset(filepath, keys: List[str]):
    with h5py.File(filepath, "r") as f:
        ret = []
        for k in keys:
            ret.append(f[k][:])
    return ret


def write_hdf5_dataset(output_path, data_dict: dict):
    with h5py.File(output_path, "w") as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v, compression="lzf")


def write_build_time(index_path: str, seconds: int):
    with open(f"{index_path}.build.seconds.txt", "w") as f:
        f.write(f"{seconds}")
