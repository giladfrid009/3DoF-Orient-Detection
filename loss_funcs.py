import abc
import numpy as np
import skimage.metrics as metrics
from utils.image import ImageUtils
import skimage
from skimage import feature
from skimage import color
from skimage import filters


class LossFunc(abc.ABC):
    def __call__(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        """
        Calculate the loss metric between two images. The more similar the images, the lower the loss value.

        Args:
            image_truth (np.ndarray): The ground truth image.
            image_other (np.ndarray): The other image to compare with.

        Returns:
            float: The calculated loss value.
        """
        assert image_truth.shape[-1] == 3
        assert image_truth.ndim == image_other.ndim == 3

        if image_truth.shape != image_other.shape:
            pad_shape = np.maximum(image_truth.shape, image_other.shape)
            image_truth = ImageUtils.pad_to_shape(image_truth, pad_shape, pad_value=0)
            image_other = ImageUtils.pad_to_shape(image_other, pad_shape, pad_value=0)

        image_truth = image_truth.astype(np.uint8, copy=False)
        image_other = image_other.astype(np.uint8, copy=False)

        return self._calculate(image_truth, image_other)

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

    def get_name(self) -> str:
        return type(self).__name__


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

    def get_name(self) -> str:
        return f"{self.inner_loss_func.get_name()}({type(self).__name__})"


class Gaussian(LossFunc):
    def __init__(self, inner_loss_func: LossFunc, sigma: int = 1):
        self.inner_loss_func = inner_loss_func
        self.sigma = sigma

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        image_other = filters.gaussian(image_other, sigma=self.sigma, channel_axis=-1)
        image_other = skimage.util.img_as_ubyte(image_other)

        return self.inner_loss_func(image_truth, image_other)

    def get_name(self) -> str:
        return f"{self.inner_loss_func.get_name()}({type(self).__name__})"


class WeightedSum(LossFunc):
    def __init__(self, loss1: LossFunc, loss2: LossFunc, w1: float = 0.5, w2: float = 0.5) -> None:
        self.loss1 = loss1
        self.loss2 = loss2
        self.w1 = w1
        self.w2 = w2

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        loss1 = self.loss1._calculate(image_truth, image_other)
        loss2 = self.loss2._calculate(image_truth, image_other)
        return self.w1 * loss1 + self.w2 * loss2

    def get_name(self) -> str:
        return f"{type(self).__name__}({self.loss1.get_name()},{self.loss2.get_name()})"


class IOU(LossFunc):
    def __init__(self, bg_value: int = 0):
        self.bg_value = bg_value

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        mask1 = ImageUtils.calc_mask(image_truth, bg_value=self.bg_value)
        mask2 = ImageUtils.calc_mask(image_other, bg_value=self.bg_value)
        iou = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
        return 1 - iou


class MSE(LossFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        image_truth = image_truth.astype(np.float64)
        image_other = image_other.astype(np.float64)
        mse = np.mean((image_truth - image_other) ** 2, dtype=np.float64)
        return mse


class RMSE(LossFunc):
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


class NMI(LossFunc):
    def __init__(self, bins: int = 50):
        self.bins = bins

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        nmi = metrics.normalized_mutual_information(image_truth, image_other, bins=self.bins)
        return 2 - nmi


class PSNR(LossFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        psnr = metrics.peak_signal_noise_ratio(image_truth, image_other, data_range=255)
        return -1 * psnr


class SSIM(LossFunc):
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


class Hausdorff(LossFunc):
    def __init__(self, bg_value: int = 0, method: str = "modified"):
        self.bg_value = bg_value
        self.method = method.lower()

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        mask1 = ImageUtils.calc_mask(image_truth, bg_value=self.bg_value)
        mask2 = ImageUtils.calc_mask(image_other, bg_value=self.bg_value)
        return metrics.hausdorff_distance(mask1, mask2, method=self.method)


class ARE(LossFunc):
    def __init__(self, quant_level: int = 1) -> None:
        assert quant_level > 0
        self.quant_level = quant_level

    def quantize(self, image: np.ndarray, levels: int) -> np.ndarray:
        if levels == 1:
            return (image > 0).astype(np.uint8, copy=False)

        bins = np.concatenate([[0], np.linspace(1, 255, levels, endpoint=True)])
        return np.digitize(image, bins)

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        quant_truth = self.quantize(image_truth, self.quant_level)
        quant_other = self.quantize(image_other, self.quant_level)
        table = metrics.contingency_table(quant_truth, quant_other, ignore_labels=None)
        error, _, _ = metrics.adapted_rand_error(quant_truth, quant_other, table=table, ignore_labels=None)
        return error


class VI(LossFunc):
    def __init__(self, quant_level: int = 1) -> None:
        assert quant_level > 0
        self.quant_level = quant_level

    def quantize(self, image: np.ndarray, levels: int) -> np.ndarray:
        if levels == 1:
            return (image > 0).astype(np.uint8, copy=False)

        bins = np.concatenate([[0], np.linspace(1, 255, levels, endpoint=True)])
        return np.digitize(image, bins)

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        quant_truth = self.quantize(image_truth, self.quant_level)
        quant_other = self.quantize(image_other, self.quant_level)
        table = metrics.contingency_table(quant_truth, quant_other, ignore_labels=None)
        h1, h2 = metrics.variation_of_information(quant_truth, quant_other, table=table, ignore_labels=None)
        return h2
