from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

type AnyArray = NDArray[Any]
type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type SplitIndices = tuple[IntArray, IntArray]
type SplitMapping = Mapping[int, SplitIndices]
type ModelKwargs = dict[str, Any]


def as_any_array(value: Any, *, dtype: Any | None = None) -> AnyArray:
    return np.asarray(value, dtype=dtype)


def as_float_array(value: Any) -> FloatArray:
    return cast(FloatArray, np.asarray(value, dtype=np.float64))


def as_int_array(value: Any) -> IntArray:
    return cast(IntArray, np.asarray(value, dtype=np.int64))


def as_bool_array(value: Any) -> BoolArray:
    return cast(BoolArray, np.asarray(value, dtype=np.bool_))
