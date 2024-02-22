import abc
import numpy as np
import skimage.metrics as metrics
from image_helpers import ImageHelpers


class MetricFunc(abc.ABC):
    def __call__(self, image1: np.ndarray, image2: np.ndarray, is_batch: bool = False) -> float | list[float]:
        assert image1.shape == image2.shape
        if is_batch:
            return self._calculate_batch(image1, image2)
        return self._calculate(image1, image2)

    @abc.abstractclassmethod
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        pass

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        scores = []
        for img_truth, img_other in zip(batch_truth, batch_other):
            score = self._calculate(img_truth, img_other)
            scores.append(score)
        return scores


class IOU(MetricFunc):
    def __init__(self, bg_value: int = 0):
        self.bg_value = bg_value

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        mask1 = ImageHelpers.calc_mask(image_truth, bg_value=self.bg_value, orig_dims=False)
        mask2 = ImageHelpers.calc_mask(image_other, bg_value=self.bg_value, orig_dims=False)
        iou = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
        score = 1 - iou
        return score

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        masks1 = ImageHelpers.calc_mask(batch_truth, bg_value=self.bg_value, orig_dims=False)
        masks2 = ImageHelpers.calc_mask(batch_other, bg_value=self.bg_value, orig_dims=False)
        N = batch_truth.shape[0]
        both = np.sum((masks1 & masks2).reshape(N, -1), axis=-1)
        any = np.sum((masks1 | masks2).reshape(N, -1), axis=-1)
        scores = 1 - both / any
        return scores.tolist()


class IntersectionMetric(MetricFunc):
    def __init__(self, metric: MetricFunc, bg_value: int = 0) -> None:
        self.metric = metric
        self.bg_value = bg_value

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        mask1 = ImageHelpers.calc_mask(image_truth, bg_value=self.bg_value, orig_dims=True)
        mask2 = ImageHelpers.calc_mask(image_other, bg_value=self.bg_value, orig_dims=True)
        intersection = mask1 & mask2
        image_truth = image_truth * intersection
        image_other = image_other * intersection
        return self.metric._calculate(image_truth, image_other)

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        mask1 = ImageHelpers.calc_mask(batch_truth, bg_value=self.bg_value, orig_dims=True)
        mask2 = ImageHelpers.calc_mask(batch_other, bg_value=self.bg_value, orig_dims=True)
        intersection = mask1 & mask2
        batch_truth = batch_truth * intersection
        batch_other = batch_other * intersection
        return self.metric._calculate(batch_truth, batch_other)


class Mean_Squared_Error(MetricFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        return metrics.mean_squared_error(image_truth, image_other)


class Normalized_Root_MSE(MetricFunc):
    def __init__(self, norm: str = "min-max"):
        self.norm = norm

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        return metrics.normalized_root_mse(image_truth, image_other, normalization=self.norm)


class Normalized_Mutual_Information(MetricFunc):
    def __init__(self, bins: int = 100):
        self.bins = bins

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        nmi = metrics.normalized_mutual_information(image_truth, image_other, bins=self.bins)
        return 2 - nmi


class Peak_Signal_Noise_Ratio(MetricFunc):
    def __init__(self, data_range: float = 255):
        self.data_range = data_range

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        return metrics.peak_signal_noise_ratio(image_truth, image_other, data_range=self.data_range)


class Structural_Similarity(MetricFunc):
    def __init__(self, data_range: float = 255, channel_axis: int = 2, win_size: int = None):
        self.data_range = data_range
        self.channel_axis = channel_axis
        self.win_size = win_size

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        return metrics.structural_similarity(
            image_truth,
            image_other,
            win_size=self.win_size,
            channel_axis=self.channel_axis,
            data_range=self.data_range,
            gradient=False,
            full=False,
        )


class Hausdorff_Distance(MetricFunc):
    def __init__(self, bg_value: int = 0, method: str = "modified"):
        self.bg_value = bg_value
        self.method = method

    def _calculate_batch(self, batch_truth: np.ndarray, batch_other: np.ndarray) -> list[float]:
        masks1 = ImageHelpers.calc_mask(batch_truth, bg_value=self.bg_value, orig_dims=False)
        masks2 = ImageHelpers.calc_mask(batch_other, bg_value=self.bg_value, orig_dims=False)
        scores = []
        for m1, m2 in zip(masks1, masks2):
            score = metrics.hausdorff_distance(m1, m2, method=self.method)
            scores.append(score)
        return scores

    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        mask1 = ImageHelpers.calc_mask(image_truth, bg_value=self.bg_value, orig_dims=False)
        mask2 = ImageHelpers.calc_mask(image_other, bg_value=self.bg_value, orig_dims=False)
        return metrics.hausdorff_distance(mask1, mask2, method=self.method)


class Adapted_Rand_Error(MetricFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        error, _, _ = metrics.adapted_rand_error(image_truth, image_other)
        fscore = 1 - error
        return fscore


class Variation_Of_Information(MetricFunc):
    def _calculate(self, image_truth: np.ndarray, image_other: np.ndarray) -> float:
        h1, h2 = metrics.variation_of_information(image_truth, image_other)
        return 1 / (h2 + 1)
