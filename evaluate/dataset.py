from __future__ import annotations
from typing import Iterator, Iterable

from manipulated_object import ObjectPosition


class EvalDataset:
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
