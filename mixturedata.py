from typing import (
    Collection,
    Generic,
    TypeVar,
    Union,
    Sequence,
    Any,
    List,
)

# from singledispatchmethod import singledispatchmethod  # pip install
from dataclasses import dataclass
import numpy as np

T = TypeVar("T")

@dataclass
class MixtureParameters(Generic[T]):
    __slots__ = ["weights", "components"]
    weights: np.ndarray
    components: Sequence[T]
