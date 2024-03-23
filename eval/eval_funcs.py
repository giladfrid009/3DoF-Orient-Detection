from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from utils.image import ImageUtils
from view_sampler import ViewSampler
from utils.orient import OrientUtils


class EvalFunc(ABC):
    def __call__(self, depth_truth: np.ndarray, depth_other: np.ndarray) -> float:
        """
        Calculate the loss metric between two depth maps. The more similar the images, the lower the loss value.

        Args:
            depth_truth (np.ndarray): The ground truth depth map.
            depth_other (np.ndarray): The other depth map to compare with.

        Returns:
            float: The calculated loss value.
        """
        assert depth_truth.shape[-1] == 1
        assert depth_truth.ndim == depth_other.ndim == 3

        if depth_truth.shape != depth_other.shape:
            pad_shape = np.maximum(depth_truth.shape, depth_other.shape)
            depth_truth = ImageUtils.pad_to_shape(depth_truth, pad_shape, pad_value=0)
            depth_other = ImageUtils.pad_to_shape(depth_other, pad_shape, pad_value=0)

        depth_truth = depth_truth.astype(np.float64, copy=False)
        depth_other = depth_other.astype(np.float64, copy=False)

        return self._calculate(depth_truth, depth_other)

    @abstractmethod
    def _calculate(self, depth_truth: np.ndarray, depth_other: np.ndarray) -> float:
        """
        Calculate the loss metric between two depth maps. The more similar the maps are, the lower the loss value.

        Args:
            depth_truth (np.ndarray): The ground truth depth.
            depth_other (np.ndarray): The other depth map to compare with.

        Returns:
            float: The calculated loss value.
        """
        pass

    def get_name(self) -> str:
        return type(self).__name__


def calculate_penalty(depth_viewer: ViewSampler, num_samples: int = 1000, seed: int = None) -> float:
    positions1 = OrientUtils.generate_random(num_samples, seed)
    positions2 = OrientUtils.generate_random(num_samples, seed)

    max_list = []
    for pos1, pos2 in zip(positions1, positions2):
        depth1 = depth_viewer.get_view_cropped(pos1, depth=True)
        depth2 = depth_viewer.get_view_cropped(pos2, depth=True)
        pad_shape = np.maximum(depth1.shape, depth2.shape)
        depth1 = ImageUtils.pad_to_shape(depth1, pad_shape, pad_value=0)
        depth2 = ImageUtils.pad_to_shape(depth2, pad_shape, pad_value=0)
        both = (depth1 > 0) & (depth2 > 0)
        max_list.append(np.max(np.abs(depth1[both] - depth2[both])))

    penalty = np.mean(max_list)
    return penalty


class XorDiff(EvalFunc):
    def __init__(self, penalty: float, p_norm: float = 1.0):
        self.penalty = penalty
        self.p_norm = p_norm

    def _calculate(self, depth_truth: np.ndarray, depth_other: np.ndarray) -> float:
        mask_truth = depth_truth > 0
        mask_other = depth_other > 0
        both_appear = mask_truth & mask_other
        one_appears = mask_truth ^ mask_other

        diffs = both_appear * (depth_truth - depth_other) + self.penalty * one_appears
        diffs = np.abs(diffs)

        union_area = np.sum(mask_truth | mask_other)
        norm = np.sum(np.power(diffs, self.p_norm))
        loss = np.power(norm, 1 / self.p_norm) / (union_area * self.penalty)
        return loss
