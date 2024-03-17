import numpy as np
import skimage


class ImageUtils:
    """
    Image manipulation functions that work both on batched data and single images.
    Image batch dim is assumed to be [N, W, H, 3]
    Single image dim is assumed to be [W, H, 3]
    """

    @staticmethod
    def depth2rgb(img: np.ndarray) -> np.ndarray:
        if img.shape[-1] == 1:
            img = img.reshape(img.shape[0], img.shape[1])
        img[img <= 0] = np.nan
        max_depth = np.nanmax(img)
        min_depth = np.nanmin(img)
        min_color, max_color = 20, 250
        img = (img - min_depth) / (max_depth - min_depth)
        img = (img * (max_color - min_color)) + min_color
        img[img == np.nan] = 0
        img = skimage.color.gray2rgb(img)
        return img.astype(np.uint8)

    @staticmethod
    def calc_mask(images: np.ndarray, bg_value: int = 0, orig_dims: bool = False) -> np.ndarray:
        mask = np.any(images != bg_value, axis=-1)
        if orig_dims:
            mask = np.broadcast_to(np.expand_dims(mask, axis=-1), images.shape)
        return mask

    @staticmethod
    def calc_bboxes(mask_batch: np.ndarray, margin_factor: float = 1.2) -> np.ndarray:
        x = np.any(mask_batch, axis=-1)
        y = np.any(mask_batch, axis=-2)

        def argmin_argmax(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # find smallest and largest indices
            imin = np.argmax(arr, axis=-1)
            arr = np.flip(arr, axis=-1)
            length = arr.shape[-1]
            imax = length - np.argmax(arr, axis=-1)

            # add margin to the indices
            diff = imax - imin
            margin = (diff * (margin_factor - 1)).astype(imin.dtype)
            imin = imin - margin
            imax = imax + margin

            # make sure we're within bounds
            imin = np.maximum(imin, 0)
            imax = np.minimum(imax, length - 1)
            return imin, imax

        xmin, xmax = argmin_argmax(x)
        ymin, ymax = argmin_argmax(y)

        return np.stack((xmin, ymin, xmax, ymax), axis=-1)

    @staticmethod
    def pad_to_shape(image: np.ndarray, shape: np.ndarray, pad_value: float = 0) -> np.ndarray:
        pad_dims = []
        for width in np.subtract(shape, image.shape):
            pad_dims.append((width // 2, width // 2 + width % 2))
        padded = np.pad(image, pad_dims, mode="constant", constant_values=pad_value)
        return padded
