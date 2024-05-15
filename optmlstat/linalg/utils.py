"""
linear algebra utils
"""

from functools import reduce

import numpy as np
from numpy import random as nr
from scipy.stats import ortho_group


def generic_array_mul(*args: np.ndarray) -> np.ndarray:
    mul_array: np.ndarray = reduce(
        np.ndarray.__mul__,
        [
            array.reshape(
                array.shape[:-1]
                + tuple([1] * idx)
                + array.shape[-1:]
                + tuple([1] * (len(args) - idx - 1))
            )
            for idx, array in enumerate(args)
        ],
    )
    return mul_array.sum(axis=tuple(range(mul_array.ndim - len(args))))


def idx_iterator(shape: tuple[int, ...]) -> list[tuple[int, ...]]:

    rev_shape: tuple[int, ...] = shape[::-1]

    return [
        tuple(
            [
                int(idx / int(np.array(rev_shape[:dim_idx]).prod())) % dim
                for dim_idx, dim in enumerate(rev_shape)
            ]
        )[::-1]
        for idx in range(int(np.array(rev_shape).prod()))
    ]


def block_array(blk_list: list) -> np.ndarray:

    shape, dims = _block_array_shape_and_dims(blk_list)

    blk_list = blk_list.copy()

    for idx_tuple in idx_iterator(shape):
        _blk_list = blk_list
        for idx in idx_tuple[:-1]:
            _blk_list = _blk_list[idx]

        _blk_list[idx_tuple[-1]] = _blk_list[idx_tuple[-1]] + np.zeros(
            [dims[_idx][idx] for _idx, idx in enumerate(idx_tuple)]
        )

    return merge_blk_array(blk_list, 0)


def merge_blk_array(blk: list | np.ndarray, axis: int, /) -> np.ndarray:
    if isinstance(blk, np.ndarray):
        return blk
    elif isinstance(blk, list):
        return np.concatenate([merge_blk_array(bl, axis + 1) for bl in blk], axis=axis)
    else:
        assert False, (blk, blk.__class__)


def _block_array_shape_and_dims(blk_list: list) -> tuple[tuple[int, ...], list[list[int]]]:
    """
    return block matrix
    """
    _blk_list: list = blk_list

    _shape: list[int] = []
    while isinstance(_blk_list, list):
        _shape.append(len(_blk_list))
        _blk_list = _blk_list[0]
    shape: tuple[int, ...] = tuple(_shape)

    dims: list[list[int]] = [list([1] * num) for num in _shape]

    for idx_tuple in idx_iterator(shape):
        _blk_list = blk_list
        for idx in idx_tuple:
            _blk_list = _blk_list[idx]

        assert isinstance(_blk_list, (np.ndarray, float, int)), (_blk_list, _blk_list.__class__)

        array: np.ndarray = np.array(_blk_list)

        for idx, size in enumerate(tuple([1] * (len(dims) - len(array.shape))) + array.shape):
            assert (
                dims[idx][idx_tuple[idx]] == 1 or size == 1 or dims[idx][idx_tuple[idx]] == size
            ), ("block array dimension mismatch", idx_tuple, dims[idx][idx_tuple[idx]], size)
            dims[idx][idx_tuple[idx]] = max(dims[idx][idx_tuple[idx]], size)

    return shape, dims


def get_random_pos_def_array(size_or_array_1d: int | np.ndarray) -> np.ndarray:
    res: np.ndarray
    if isinstance(size_or_array_1d, int):
        r_array_2d: np.ndarray = nr.randn(size_or_array_1d, size_or_array_1d)
        res = np.dot(r_array_2d, r_array_2d.T)
    elif isinstance(size_or_array_1d, np.ndarray):
        assert size_or_array_1d.ndim == 1, size_or_array_1d.shape
        orth_array_2d: np.ndarray = ortho_group.rvs(dim=size_or_array_1d.size)
        return np.dot(orth_array_2d, np.dot(np.diag(size_or_array_1d), orth_array_2d.T))
    else:
        assert False, size_or_array_1d.__class__

    return res


def get_random_orthogonal_array(num: int) -> np.ndarray:
    """
    name comes from linear algebra term, "orthogonal matrix",
    which is a matrix whose column (or equivalently row) vectors
    are ortho*normal*!
    """
    return ortho_group.rvs(dim=num)
