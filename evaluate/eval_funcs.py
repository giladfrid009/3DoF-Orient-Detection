from abc import ABC, abstractmethod
import numpy as np
from utils.image import ImageUtils


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


class XorDiff(EvalFunc):
    def __init__(self, penalty: float, p_norm: float = 1.0):
        self.obj_depth = penalty
        self.p_norm = p_norm

    def _calculate(self, depth_truth: np.ndarray, depth_other: np.ndarray) -> float:
        mask_truth = depth_truth > 0
        mask_other = depth_other > 0
        both_appear = mask_truth & mask_other
        one_appears = mask_truth ^ mask_other

        diffs = np.zeros_like(depth_truth, dtype=np.float64)
        diffs += both_appear * (depth_truth - depth_other)
        diffs += self.obj_depth * one_appears
        diffs = np.abs(diffs)

        n = np.sum(mask_truth | mask_other)
        loss = np.power(np.sum(np.power(diffs, self.p_norm)) / n, 1 / self.p_norm)
        return loss

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        masks_truth = batch_truth > 0
        masks_other = batch_other > 0
        both_appear = masks_truth & masks_other
        one_appears = masks_truth ^ masks_other

        diffs = np.zeros_like(batch_truth, dtype=np.float64)
        diffs += both_appear * (batch_truth - batch_other)
        diffs += self.obj_depth * one_appears
        diffs = np.abs(diffs)

        n = np.sum(batch_truth | batch_other, axis=(1, 2, 3))
        losses = np.power(np.sum(np.power(diffs, self.p_norm), axis=(1, 2, 3)) / n, 1 / self.p_norm)
        return losses.tolist()


class NormXorDiff(EvalFunc):
    def __init__(self, penalty: float, p_norm: float = 1.0, normalization: str = "euclidean"):
        self.penalty = penalty
        self.p_norm = p_norm
        self.norm = normalization.lower()
        assert self.norm in ["euclidean", "min-max", "mean"]

    def _calculate(self, depth_truth: np.ndarray, depth_other: np.ndarray) -> float:
        mask_truth = depth_truth > 0
        mask_other = depth_other > 0
        both_appear = mask_truth & mask_other
        one_appears = mask_truth ^ mask_other

        diffs = np.zeros_like(depth_truth, dtype=np.float64)
        diffs += both_appear * (depth_truth - depth_other)
        diffs += self.penalty * one_appears
        diffs = np.abs(diffs)

        n = np.sum(mask_truth | mask_other)
        loss = np.sum(np.power(diffs, self.p_norm)) / n
        loss = np.power(loss, 1 / self.p_norm)

        if self.norm == "euclidean":
            denom = np.sqrt(np.mean(depth_truth**2, dtype=np.float64))
        elif self.norm == "min-max":
            denom = depth_truth.max() - depth_truth.min()
        elif self.norm == "mean":
            denom = depth_truth.mean()

        normalized = loss / denom
        return normalized

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        masks_truth = batch_truth > 0
        masks_other = batch_other > 0
        both_appear = masks_truth & masks_other
        one_appears = masks_truth ^ masks_other

        diffs = np.zeros_like(batch_truth, dtype=np.float64)
        diffs += both_appear * (batch_truth - batch_other)
        diffs += self.penalty * one_appears
        diffs = np.abs(diffs)

        n = np.sum(batch_truth | batch_other, axis=(1, 2, 3))
        losses = np.sum(np.power(diffs, self.p_norm), axis=(1, 2, 3)) / n
        losses = np.power(losses, 1 / self.p_norm)

        if self.norm == "euclidean":
            denom = np.sqrt(np.mean(batch_truth**2, axis=(1, 2, 3), dtype=np.float64))
        elif self.norm == "min-max":
            denom = np.max(batch_truth, axis=(1, 2, 3)) - np.min(batch_truth, axis=(1, 2, 3))
        elif self.norm == "mean":
            denom = np.mean(batch_truth, axis=(1, 2, 3))

        normalized = losses / denom
        return normalized.tolist()
