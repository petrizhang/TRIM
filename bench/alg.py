from typing import Any, Dict

import numpy


class BaseANN(object):
    """Base class/interface for Approximate Nearest Neighbors (ANN) algorithms used in benchmarking."""

    def fit(self, X: numpy.array) -> None:
        """Fits the ANN algorithm to the provided data. 

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            X (numpy.array): The data to fit the algorithm to.
        """
        pass

    def ann_query(self, q: numpy.array, n: int) -> numpy.array:
        """Performs a query on the algorithm to find the nearest neighbors. 

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            q (numpy.array): The vector to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return.

        Returns:
            numpy.array: An array of indices representing the nearest neighbors.
        """
        return []  # array of candidate indices
    
    def range_query(self, q: numpy.array, r: float) -> numpy.array:

        return []  # array of candidate indices

    def get_additional(self) -> Dict[str, Any]:
        """Returns additional attributes to be stored with the result.

        Returns:
            dict: A dictionary of additional attributes.
        """
        return {}
    
    def get_pruning_ratio():
        pass
    
    def set_query_arguments(self, **kwargs):
        pass

    def set_data(self, base_data):
        pass

    def save_index(self, index_path: str) -> None:
        pass

    def load_index(self, index_path: str) -> None:
        pass

    def __str__(self) -> str:
        return self.name
