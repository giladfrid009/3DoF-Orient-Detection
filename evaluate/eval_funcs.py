from abc import ABC, abstractmethod
import numpy as np


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
        assert depth_truth.shape == depth_other.shape

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


class IOU_Diff(EvalFunc):
    def __init__(self, bg_depth: float = 20, max_depth: float = 1.0, method: str = "mae"):
        self.bg_depth = bg_depth
        self.max_depth = max_depth
        self.method = method.lower()
        assert self.method in ["mae", "mse"]

    def _calculate(self, depth_truth: np.ndarray, depth_other: np.ndarray) -> float:
        mask1 = depth_truth < self.bg_depth
        mask2 = depth_other < self.bg_depth

        both_appear = mask1 & mask2
        one_appears = mask1 ^ mask2

        diffs = np.zeros_like(depth_truth, dtype=np.float64)
        diffs += self.max_depth * one_appears

        if self.method == "mae":
            diffs += np.abs(both_appear * (depth_truth - depth_other))
        elif self.method == "mse":
            diffs += (both_appear * (depth_truth - depth_other)) ** 2

        return np.mean(diffs)

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        masks1 = batch_truth < self.bg_depth
        masks2 = batch_other < self.bg_depth

        both_appear = masks1 & masks2
        one_appears = masks1 ^ masks2

        diffs = np.zeros_like(batch_truth, dtype=np.float64)
        diffs += self.max_depth * one_appears
        diffs += np.abs(both_appear * (batch_truth - batch_other))

        if self.method == "mae":
            diffs += np.abs(both_appear * (batch_truth - batch_other))
        elif self.method == "mse":
            diffs += (both_appear * (batch_truth - batch_other)) ** 2

        return np.mean(diffs, axis=(1, 2, 3))
