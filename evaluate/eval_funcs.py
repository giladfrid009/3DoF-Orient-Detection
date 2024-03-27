from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from utils.image import ImageUtils
from view_sampler import ViewSampler
from utils.orient import OrientUtils
from manipulated_object import ObjectPosition


class EvalFunc(ABC):
    def __call__(self, depth_truth: np.ndarray, depth_other: np.ndarray) -> float | list[float]:
        """
        Calculate the loss metric between two depth maps. The more similar the images, the lower the loss value.

        Args:
            depth_truth (np.ndarray): The ground truth depth map.
            depth_other (np.ndarray): The other depth map to compare with.

        Returns:
            float | list[float]: The calculated loss value(s).
        """
        assert depth_truth.shape[-1] == 1
        assert depth_truth.ndim == depth_other.ndim

        if depth_truth.shape != depth_other.shape:
            pad_shape = np.maximum(depth_truth.shape, depth_other.shape)
            depth_truth = ImageUtils.pad_to_shape(depth_truth, pad_shape, pad_value=0)
            depth_other = ImageUtils.pad_to_shape(depth_other, pad_shape, pad_value=0)

        depth_truth = depth_truth.astype(np.float64, copy=False)
        depth_other = depth_other.astype(np.float64, copy=False)

        if depth_truth.ndim == 4:
            return self._calculate_batch(depth_truth, depth_other)

        if depth_truth.ndim == 3:
            return self._calculate(depth_truth, depth_other)

        raise ValueError("Invalid input image shape. Must be either [W, H, 1] or [N, W, H, 1]")

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

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        """
        Calculate the loss metric between two batches of depth maps. The more similar the maps are, the lower the loss value.

        Args:
            batch_truth (np.ndarray): The batch of ground truth depths.
            batch_other (np.ndarray): The batch of other depth maps to compare with.

        Returns:
            list[float]: The calculated loss values for each pair of depth maps in the batch.
        """
        losses = []
        for img_truth, img_other in zip(batch_truth, batch_other):
            loss = self._calculate(img_truth, img_other)
            losses.append(loss)
        return losses

    def get_name(self) -> str:
        return type(self).__name__

def generate_positions(count: int, location:tuple[float,float,float], seed:int=None) -> list[ObjectPosition]:
    orients = OrientUtils.generate_random(count, seed)
    positions = [ObjectPosition(orient, location) for orient in orients]
    return positions

def calculate_penalty(depth_viewer: ViewSampler, num_samples: int, location:tuple[float,float,float], seed: int = None) -> float:
    positions1 = generate_positions(num_samples, location, seed)
    positions2 = generate_positions(num_samples, location, seed+1)

    total = 0
    count = 0
    losses = []
    for pos1, pos2 in zip(positions1, positions2):
        depth1, _ = depth_viewer.get_view_cropped(pos1, depth=True)
        depth2, _ = depth_viewer.get_view_cropped(pos2, depth=True)
        pad_shape = np.maximum(depth1.shape, depth2.shape)
        depth1 = ImageUtils.pad_to_shape(depth1, pad_shape, pad_value=0)
        depth2 = ImageUtils.pad_to_shape(depth2, pad_shape, pad_value=0)
        both = (depth1 > 0) & (depth2 > 0)
        # total += np.sum(np.abs(depth1[both] - depth2[both]))
        # count += np.sum(both)
        if np.sum(both) == 0:
            losses.append(0)
            continue
        losses.append(np.max(np.abs(depth1[both] - depth2[both])))

    # penalty = total / count
    penalty = np.mean(losses)
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

        diffs = np.zeros_like(depth_truth, dtype=np.float64)
        diffs += both_appear * (depth_truth - depth_other)
        diffs += self.penalty * one_appears
        diffs = np.abs(diffs)

        union_area = np.sum(mask_truth | mask_other)
        if union_area == 0:
            print(f"ERROR::XorDiff::union_area = {union_area}")
        norm = np.sum(np.power(diffs, self.p_norm))
        loss = np.power(norm, 1 / self.p_norm) / (self.penalty * union_area)
        return loss
