import cv2
import numpy as np


def transform_points(points, matrix):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-8):
        raise ValueError('Point projects to infinity')
    return (homogeneous[:, :2] / homogeneous[:, 2:]).astype(np.float32)


def letterbox(image, size=192):
    """Exact documented pixel-center affine; keep its inverse for decoding."""
    h, w = image.shape[:2]
    scale = size / max(w, h)
    tx = (size - 1) / 2 - scale * (w - 1) / 2
    ty = (size - 1) / 2 - scale * (h - 1) / 2
    matrix = np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1]], np.float64)
    result = cv2.warpAffine(image, matrix[:2], (size, size),
                            flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    return result, matrix


def network_to_heatmap(points, stride=4):
    return (np.asarray(points) + 0.5) / stride - 0.5


def heatmap_to_network(points, stride=4):
    return (np.asarray(points) + 0.5) * stride - 0.5


def center_offsets(points, width, height):
    return np.asarray(points) - np.array([(width - 1) / 2, (height - 1) / 2])
