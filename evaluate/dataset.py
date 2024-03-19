from __future__ import annotations
from typing import Iterator, Iterable
import math
import numpy as np
import cv2 as cv

from view_sampler import ViewSampler
from utils.orient import OrientUtils
from utils.io import save_pickle, load_pickle
from manipulated_object import ObjectPosition
from utils.image import ImageUtils


class Dataset:
    @staticmethod
    def create_uniform(location: tuple[float, float, float], min_samples: int) -> Dataset:
        orients = OrientUtils.generate_uniform(min_samples)
        return Dataset([ObjectPosition(orient, location) for orient in orients])

    @staticmethod
    def create_random(location: tuple[float, float, float], num_samples: int, seed: int = None) -> Dataset:
        orients = OrientUtils.generate_random(num_samples, seed)
        return Dataset([ObjectPosition(orient, location) for orient in orients])

    @staticmethod
    def load(path: str) -> Dataset:
        return load_pickle(path)

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

    def extend(self, other: Dataset):
        self._items.extend(other._items)

    def save(self, path: str):
        save_pickle(path, self)

    def visualize(
        self,
        view_sampler: ViewSampler,
        depth: bool = False,
        batch_size: int = 9,
    ):
        nrows = math.ceil(math.sqrt(batch_size))
        ncols = math.ceil(batch_size / nrows)

        for i in range(0, len(self), batch_size):
            pos_batch = self[i : i + batch_size]

            images = [view_sampler.get_view_cropped(pos, depth=depth)[0] for pos in pos_batch]
            if depth:
                images = [ImageUtils.depth2rgb(img) for img in images]

            shapes = np.asanyarray([img.shape for img in images])
            max_shape = shapes.max(axis=0)
            images = [ImageUtils.pad_to_shape(img, max_shape) for img in images]

            for j, img in enumerate(images):
                cv.putText(
                    img,
                    "img #{}".format(i + j),
                    (0, 12),
                    cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=2,
                )

            # make sure array of images is exactly nrows x ncols
            if len(images) < batch_size:
                images.extend([np.zeros_like(images[0])] * (batch_size - len(images)))

            img_mat = [np.hstack(images[i * ncols : (i + 1) * ncols]) for i in range(nrows)]
            cv.imshow("Dataset Visualization", np.vstack(img_mat))
            cv.waitKey(0)
