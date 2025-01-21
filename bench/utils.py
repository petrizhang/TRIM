from typing import List

import h5py
import numpy as np


def format_dict(d: dict) -> str:
    return '_'.join(f'{key}-{value}' for key, value in d.items())


def cos_normalize(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """normalization function for cosine space datasets
    Args:
        x (np.ndarray): the vectors
        min_value (float): this value is fully ignored
        max_value (float): this value is fully ignored
    Returns:
        np.ndarray: [description]
    """
    norm = np.linalg.norm(x, axis=1)
    norm.resize((len(norm), 1))
    ret = x / norm
    # set elements of all-zero vectors as a large-enough number (here 100)
    # thus they cannot become KNN of any given queries
    ret[np.isnan(ret)] = 100
    return ret


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
