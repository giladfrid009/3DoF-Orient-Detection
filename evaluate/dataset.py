from __future__ import annotations
from typing import Iterator, Iterable

from utils.orient_helpers import OrientUtils
from manipulated_object import ObjectPosition


class EvalDataset:
    @staticmethod
    def create_uniform(location: tuple[float, float, float], min_samples: int) -> EvalDataset:
        orients = OrientUtils.generate_uniform(min_samples)
        return EvalDataset([ObjectPosition(orient, location) for orient in orients])

    def create_random(location: tuple[float, float, float], min_samples: int, rnd_seed: int = None) -> EvalDataset:
        orients = OrientUtils.generate_random(min_samples, rnd_seed)
        return EvalDataset([ObjectPosition(orient, location) for orient in orients])

    def __init__(self, items: Iterable[ObjectPosition] = None) -> None:
        self._items = list(items) if items is not None else []

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx: int) -> ObjectPosition:
        return self._items[idx]

    def __iter__(self) -> Iterator[ObjectPosition]:
        return self._items.__iter__()

    def add_item(self, orient: ObjectPosition):
        self._items.append(orient)

    def remove_item(self, idx: int):
        self._items.pop(idx)

    def extend(self, other: EvalDataset):
        self._items.extend(other._items)
