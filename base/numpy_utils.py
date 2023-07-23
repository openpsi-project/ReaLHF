from typing import Dict

import numpy as np


def split_to_shapes(x: np.ndarray, shapes: Dict, axis: int = -1):
    """Split an array and reshape to desired shapes.

    Args:
        x (np.ndarray): The array to be splitted
        shapes (Dict): Dict of shapes (tuples) specifying how to split.
        axis (int): Split dimension.

    Returns:
        List: Splitted observations.
    """
    axis = len(x.shape) + axis if axis < 0 else axis
    split_lengths = [np.prod(shape) for shape in shapes.values()]
    assert x.shape[axis] == sum(split_lengths)
    accum_split_lengths = [sum(split_lengths[:i]) for i in range(1, len(split_lengths))]
    splitted_x = np.split(x, accum_split_lengths, axis)
    return {
        k: x.reshape(*x.shape[:axis], *shape, *x.shape[axis + 1:])
        for x, (k, shape) in zip(splitted_x, shapes.items())
    }
