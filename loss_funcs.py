import abc
import numpy as np
import skimage.metrics as metrics
from utils.image import ImageUtils
import skimage
from skimage import feature
from skimage import color
from skimage import filters


class LossFunc(abc.ABC):
    def __call__(self, image_truth: np.ndarray, image_other: np.ndarray) -> float | list[float]:
        """
        Calculate the loss metric between two images. The more similar the images, the lower the loss value.

        Args:
            image_truth (np.ndarray): The ground truth image.
            image_other (np.ndarray): The other image to compare with.

        Returns:
            float | list[float]: The calculated loss value(s).
        """
        assert image_truth.shape[-1] == 3
        assert image_truth.shape == image_other.shape
        assert image_truth.dtype == image_truth.dtype == np.uint8

        if image_truth.ndim == 4:
            return self._calculate_batch(image_truth, image_other)

        if image_truth.ndim == 3:
            return self._calculate(image_truth, image_other)

        raise ValueError("Invalid input image shape. Must be either [W, H, 3] or [N, W, H, 3]")

    @abc.abstractclassmethod
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        """
        Calculate the loss metric between two images. The more similar the images, the lower the loss value.

        Args:
            image_truth (np.ndarray): The ground truth image.
            image_other (np.ndarray): The other image to compare with.

        Returns:
            float: The calculated loss value.
        """
        pass

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        """
        Calculate the loss metric between two batches of images. The more similar the images, the lower the loss value.

        Args:
            batch_truth (np.ndarray): The batch of ground truth images.
            batch_other (np.ndarray): The batch of other images to compare with.

        Returns:
            list[float]: The calculated loss values for each pair of images in the batch.
        """
        losses = []
        for img_truth, img_other in zip(batch_truth, batch_other):
            loss = self._calculate(img_truth, img_other)
            losses.append(loss)
        return losses


class CannyEdges(LossFunc):
    def __init__(self, inner_loss_func: LossFunc, sigma: float = 1.75):
        self.inner_loss_func = inner_loss_func
        self.sigma = sigma

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        gray_truth = color.rgb2gray(image_truth)
        gray_other = color.rgb2gray(image_other)

        canny_truth = feature.canny(gray_truth, sigma=self.sigma, use_quantiles=True)
        canny_other = feature.canny(gray_other, sigma=self.sigma, use_quantiles=True)

        image_truth = np.broadcast_to(np.expand_dims(canny_truth, axis=-1), image_truth.shape)
        image_other = np.broadcast_to(np.expand_dims(canny_other, axis=-1), image_other.shape)

        image_truth = (image_truth * 255).astype(np.uint8)
        image_other = (image_other * 255).astype(np.uint8)

        return self.inner_loss_func(image_truth, image_other)

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        gray_truth = color.rgb2gray(batch_truth)
        gray_other = color.rgb2gray(batch_other)

        edges_truth = []
        edges_other = []
        for img_truth, img_other in zip(gray_truth, gray_other):
            canny_truth = feature.canny(img_truth, sigma=self.sigma)
            canny_other = feature.canny(img_other, sigma=self.sigma)

            edges_truth.append(canny_truth)
            edges_other.append(canny_other)

        edges_truth = np.stack(edges_truth, axis=0)
        edges_other = np.stack(edges_other, axis=0)

        batch_truth = np.broadcast_to(np.expand_dims(edges_truth, axis=-1), batch_truth.shape)
        batch_other = np.broadcast_to(np.expand_dims(edges_other, axis=-1), batch_other.shape)

        batch_truth = (batch_truth * 255).astype(np.uint8)
        batch_other = (batch_other * 255).astype(np.uint8)

        return self.inner_loss_func(batch_truth, batch_other)


class Gaussian(LossFunc):
    def __init__(self, inner_loss_func: LossFunc, sigma: int = 1):
        self.inner_loss_func = inner_loss_func
        self.sigma = sigma

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        image_other = filters.gaussian(image_other, sigma=self.sigma, channel_axis=-1)
        image_other = skimage.util.img_as_ubyte(image_other)

        return self.inner_loss_func(image_truth, image_other)


class IOU(LossFunc):
    def __init__(self, bg_value: int = 0):
        self.bg_value = bg_value

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        mask1 = ImageUtils.calc_mask(image_truth, bg_value=self.bg_value)
        mask2 = ImageUtils.calc_mask(image_other, bg_value=self.bg_value)
        iou = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
        return 1 - iou

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        masks1 = ImageUtils.calc_mask(batch_truth, bg_value=self.bg_value)
        masks2 = ImageUtils.calc_mask(batch_other, bg_value=self.bg_value)
        N = batch_truth.shape[0]
        both = np.sum((masks1 & masks2).reshape(N, -1), axis=-1)
        any = np.sum((masks1 | masks2).reshape(N, -1), axis=-1)
        ious = 1 - both / any
        return ious.tolist()


class MSE(LossFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        image_truth = image_truth.astype(np.float64)
        image_other = image_other.astype(np.float64)
        mse = np.mean((image_truth - image_other) ** 2, dtype=np.float64)
        return mse

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        batch_truth = batch_truth.astype(np.float64)
        batch_other = batch_other.astype(np.float64)
        mses = np.mean((batch_truth - batch_other) ** 2, axis=(1, 2, 3), dtype=np.float64)
        return mses.tolist()


class NormMSE(LossFunc):
    def __init__(self, norm: str = "euclidean"):
        self.norm = norm.lower()
        assert self.norm in ["euclidean", "min-max", "mean"]

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        image_truth = image_truth.astype(np.float64)
        image_other = image_other.astype(np.float64)
        mse = np.mean((image_truth - image_other) ** 2, dtype=np.float64)

        if self.norm == "euclidean":
            denom = np.sqrt(np.mean(image_truth**2, dtype=np.float64))
        elif self.norm == "min-max":
            denom = image_truth.max() - image_truth.min()
        elif self.norm == "mean":
            denom = image_truth.mean()

        return np.sqrt(mse) / denom

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        batch_truth = batch_truth.astype(np.float64)
        batch_other = batch_other.astype(np.float64)
        mses = np.mean((batch_truth - batch_other) ** 2, axis=(1, 2, 3))

        if self.norm == "euclidean":
            denom = np.sqrt(np.mean(batch_truth**2, axis=(1, 2, 3), dtype=np.float64))
        elif self.norm == "min-max":
            denom = np.max(batch_truth, axis=(1, 2, 3)) - np.min(batch_truth, axis=(1, 2, 3))
        elif self.norm == "mean":
            denom = np.mean(batch_truth, axis=(1, 2, 3))

        normalized = np.sqrt(mses) / denom
        return normalized.tolist()


class MutualInformation(LossFunc):
    def __init__(self, bins: int = 100):
        self.bins = bins

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        nmi = metrics.normalized_mutual_information(image_truth, image_other, bins=self.bins)
        return 2 - nmi


class PeakSignalNoiseRation(LossFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        psnr = metrics.peak_signal_noise_ratio(image_truth, image_other, data_range=255)
        return -1 * psnr


class StructuralSimilarity(LossFunc):
    def __init__(self, win_size: int = None):
        self.win_size = win_size

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        similarity = metrics.structural_similarity(
            image_truth,
            image_other,
            win_size=self.win_size,
            channel_axis=-1,
            data_range=255,
            gradient=False,
            full=False,
        )

        return 1 - similarity


class HausdorffDistance(LossFunc):
    def __init__(self, bg_value: int = 0, method: str = "modified"):
        self.bg_value = bg_value
        self.method = method.lower()

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        masks1 = ImageUtils.calc_mask(batch_truth, bg_value=self.bg_value)
        masks2 = ImageUtils.calc_mask(batch_other, bg_value=self.bg_value)
        losses = []
        for m1, m2 in zip(masks1, masks2):
            score = metrics.hausdorff_distance(m1, m2, method=self.method)
            losses.append(score)
        return losses

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        mask1 = ImageUtils.calc_mask(image_truth, bg_value=self.bg_value)
        mask2 = ImageUtils.calc_mask(image_other, bg_value=self.bg_value)
        return metrics.hausdorff_distance(mask1, mask2, method=self.method)


class AdaptedRandError(LossFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        error, _, _ = metrics.adapted_rand_error(image_truth, image_other)
        return error


class VariationOfInformation(LossFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        h1, h2 = metrics.variation_of_information(image_truth, image_other)
        return h2
