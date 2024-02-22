import abc
import numpy as np
import skimage.metrics as metrics
from image_helpers import ImageHelpers
import skimage.feature as feature
import skimage.color as color


class MetricFunc(abc.ABC):
    def __call__(self, image1: np.ndarray, image2: np.ndarray, is_batch: bool = False) -> float | list[float]:
        assert image1.shape == image2.shape
        assert image1.shape[-1] == 3
        assert image1.dtype == image1.dtype == np.uint8
        if is_batch:
            return self._calculate_batch(image1, image2)
        return self._calculate(image1, image2)

    @abc.abstractclassmethod
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        pass

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        dists = []
        for img_truth, img_other in zip(batch_truth, batch_other):
            dist = self._calculate(img_truth, img_other)
            dists.append(dist)
        return dists


class CannyEdges(MetricFunc):
    def __init__(self, inner_metric: MetricFunc, sigma: float = 1.5):
        self.inner_metric = inner_metric
        self.sigma = sigma

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        gray_truth = color.rgb2gray(image_truth)
        gray_other = color.rgb2gray(image_other)
        canny_truth = feature.canny(gray_truth, sigma=self.sigma, use_quantiles=True)
        canny_other = feature.canny(gray_other, sigma=self.sigma, use_quantiles=True)
        canny_truth = (canny_truth * 255).astype(np.uint8)
        canny_other = (canny_other * 255).astype(np.uint8)
        image_truth = np.broadcast_to(np.expand_dims(canny_truth, axis=-1), image_truth.shape)
        image_other = np.broadcast_to(np.expand_dims(canny_other, axis=-1), image_other.shape)

        return self.inner_metric(image_truth, image_other, is_batch=False)

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

        return self.inner_metric(batch_truth, batch_other, is_batch=True)


class IOU(MetricFunc):
    def __init__(self, bg_value: int = 0):
        self.bg_value = bg_value

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        mask1 = ImageHelpers.calc_mask(image_truth, bg_value=self.bg_value, orig_dims=False)
        mask2 = ImageHelpers.calc_mask(image_other, bg_value=self.bg_value, orig_dims=False)
        iou = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
        return 1 - iou

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        masks1 = ImageHelpers.calc_mask(batch_truth, bg_value=self.bg_value, orig_dims=False)
        masks2 = ImageHelpers.calc_mask(batch_other, bg_value=self.bg_value, orig_dims=False)
        N = batch_truth.shape[0]
        both = np.sum((masks1 & masks2).reshape(N, -1), axis=-1)
        any = np.sum((masks1 | masks2).reshape(N, -1), axis=-1)
        dists = 1 - both / any
        return dists.tolist()


class MSE(MetricFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        return metrics.mean_squared_error(image_truth, image_other)


class RMSE(MetricFunc):
    def __init__(self, norm: str = "min-max"):
        self.norm = norm

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        return metrics.normalized_root_mse(image_truth, image_other, normalization=self.norm)


class NMI(MetricFunc):
    def __init__(self, bins: int = 100):
        self.bins = bins

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        nmi = metrics.normalized_mutual_information(image_truth, image_other, bins=self.bins)
        return 2 - nmi


class PSNR(MetricFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        psnr = metrics.peak_signal_noise_ratio(image_truth, image_other, data_range=255)
        return -1 * psnr


class StructuralSimilarity(MetricFunc):
    def __init__(self, channel_axis: int = 2, win_size: int = None):
        self.channel_axis = channel_axis
        self.win_size = win_size

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        similarity = metrics.structural_similarity(
            image_truth,
            image_other,
            win_size=self.win_size,
            channel_axis=self.channel_axis,
            data_range=255,
            gradient=False,
            full=False,
        )

        return 1 - similarity


class HausdorffDistance(MetricFunc):
    def __init__(self, bg_value: int = 0, method: str = "modified"):
        self.bg_value = bg_value
        self.method = method

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        masks1 = ImageHelpers.calc_mask(batch_truth, bg_value=self.bg_value, orig_dims=False)
        masks2 = ImageHelpers.calc_mask(batch_other, bg_value=self.bg_value, orig_dims=False)
        dists = []
        for m1, m2 in zip(masks1, masks2):
            score = metrics.hausdorff_distance(m1, m2, method=self.method)
            dists.append(score)
        return dists

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        mask1 = ImageHelpers.calc_mask(image_truth, bg_value=self.bg_value, orig_dims=False)
        mask2 = ImageHelpers.calc_mask(image_other, bg_value=self.bg_value, orig_dims=False)
        return metrics.hausdorff_distance(mask1, mask2, method=self.method)


class AdaptedRandError(MetricFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        error, _, _ = metrics.adapted_rand_error(image_truth, image_other)
        return error


class VariationOfInformation(MetricFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        h1, h2 = metrics.variation_of_information(image_truth, image_other)
        return h2
