
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Applies the sigmoid function to each element of x.
    
    Args:
        x (np.ndarray): The numpy array to transform.
    
    Returns:
        np.ndarray: The transformed array.
    
    """
    return 1 / (1 + np.exp(-x))


def ReLU(x: np.ndarray) -> np.ndarray:
    """Applies the rectified linear unit function to each element of x.
    
    Args:
        x (np.ndarray): The numpy array to transform.
    
    Returns:
        np.ndarray: The transformed array.

    """
    return np.max(0, x)
