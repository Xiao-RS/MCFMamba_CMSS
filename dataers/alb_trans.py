import cv2
import random
import numpy as np

from .alb_core.transforms_interface import DualTransform


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, factor=0, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        if len(img.shape) == 3:
            return np.ascontiguousarray(np.rot90(img, factor, (1, 2)))
        else:
            return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        # Random int in the range [0, 3]
        return {"factor": random.randint(0, 3)}


class Flip(DualTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, code: int = 0, **params) -> np.ndarray:
        """Args:
        code (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        if code == -1:
            axis = (-1, -2)
        elif code == 0:
            axis = -1
        elif code == 1:
            axis = -2
        return np.flip(img, axis=axis)

    def get_params(self):
        # Random int in the range [-1, 1]
        return {"code": random.randint(-1, 1)}


class Vaihingen_Mirror(DualTransform):
    def apply(self, img: np.ndarray, prob: float = 0.0, **params) -> np.ndarray:

        if prob < 0.5:
            if len(img.shape) == 2:
                img = img[:, ::-1]
            else:
                img = img[:, :, ::-1]
        return img

    def get_params(self):
        return {"prob": random.random()}


class Vaihingen_Flip(DualTransform):
    def apply(self, img: np.ndarray, prob: float = 0.0, **params) -> np.ndarray:

        if prob < 0.5:
            if len(img.shape) == 2:
                img = img[::-1, :]
            else:
                img = img[:, ::-1, :]
        return img

    def get_params(self):
        return {"prob": random.random()}
