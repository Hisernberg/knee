import cv2, numpy as np

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))


def make_input(img):
    """img: uint8 HxW (1024). Returns float32 3xHxW: raw, CLAHE, background-flattened."""
    a = img.astype(np.float32) / 255.0
    c = _clahe.apply(img).astype(np.float32) / 255.0
    bg = cv2.medianBlur(img, 31).astype(np.float32) / 255.0
    flat = np.clip((a - bg) * 4 + 0.5, 0, 1)
    return np.stack([a, c, flat]).astype(np.float32)
