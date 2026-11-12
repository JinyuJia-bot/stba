"""Pixel-center coordinates for undistorted working images and dense warp grids.

All sizes are (width, height), all points end in (x, y). Pixel centers are at
integer coordinates. Normalized coordinates use grid_sample align_corners=False:
g_x = 2*(x + 0.5)/width - 1. Neither these helpers nor a resize removes distortion.
NumPy arrays and PyTorch tensors are accepted; tensor dtype/device are preserved.
"""
import numpy as np
try:
    import torch
except ImportError:  # Calibration-only and small CPU checks do not require torch.
    torch = None


def is_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor)


def as_float(value):
    if is_tensor(value):
        return value if value.is_floating_point() else value.to(torch.get_default_dtype())
    array = np.asarray(value)
    return array if np.issubdtype(array.dtype, np.floating) else array.astype(np.float64)


def as_like(value, like):
    if is_tensor(like):
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if is_tensor(value):
        raise TypeError('Mixed tensor/NumPy inputs: convert explicitly to avoid losing gradients')
    return np.asarray(value, dtype=like.dtype)


def _size(size):
    values = np.asarray(size, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(values < 1):
        raise ValueError('size must be finite positive (width, height)')
    return values


def _points(points):
    points = as_float(points)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError('pixel/grid coordinates must end in (x, y)')
    return points


def pixel_to_normalized(uv, size):
    """Integer-center pixels -> normalized grid_sample coordinates, no clamping."""
    uv = _points(uv)
    return 2 * (uv + .5) / as_like(_size(size), uv) - 1


def normalized_to_pixel(grid, size):
    """Normalized align_corners=False coordinates -> integer-center pixels."""
    grid = _points(grid)
    return (grid + 1) * as_like(_size(size), grid) / 2 - .5


def resize_pixel(uv, old_size, new_size):
    """Pixel-center mapping for a whole-image resize with no crop or padding."""
    uv = _points(uv)
    scale = as_like(_size(new_size) / _size(old_size), uv)
    return (uv + .5) * scale - .5


def resize_K(K, old_size, new_size):
    """Transform pinhole K by the same resize as resize_pixel (incl. skew)."""
    K = as_float(K)
    if K.shape[-2:] != (3, 3):
        raise ValueError('K must end in shape (3,3)')
    sx, sy = _size(new_size) / _size(old_size)
    affine = as_like([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]], K)
    return affine @ K


def crop_pixel(uv, xy_origin):
    """Crop coordinates; xy_origin is the original top-left pixel-center index."""
    uv = _points(uv)
    origin = as_like(xy_origin, uv)
    if origin.shape != (2,):
        raise ValueError('crop origin must have shape (2,)')
    return uv - origin


def crop_K(K, xy_origin):
    """Transform pinhole K for a crop, without silently changing image size."""
    K = as_float(K)
    x, y = np.asarray(xy_origin, dtype=float)
    affine = as_like([[1, 0, -x], [0, 1, -y], [0, 0, 1]], K)
    return affine @ K


def pixel_grid(size, like=None):
    """Dense HxWx2 integer-center grid. Pass a tensor as like for torch output."""
    width, height = _size(size)
    if width != int(width) or height != int(height):
        raise ValueError('grid dimensions must be integers')
    width, height = int(width), int(height)
    if is_tensor(like):
        dtype = like.dtype if like.is_floating_point() else torch.get_default_dtype()
        yy, xx = torch.meshgrid(torch.arange(height, dtype=dtype, device=like.device),
                                torch.arange(width, dtype=dtype, device=like.device), indexing='ij')
        return torch.stack((xx, yy), dim=-1)
    yy, xx = np.mgrid[:height, :width]
    dtype = as_float(like).dtype if like is not None else np.float64
    return np.stack((xx, yy), axis=-1).astype(dtype)


def inside_image(uv, size, margin=0.):
    """Validity for interpolation without any padding, optionally a patch margin."""
    uv = _points(uv)
    wh = as_like(_size(size), uv)
    if margin < 0:
        raise ValueError('margin must be nonnegative')
    finite = torch.isfinite(uv).all(dim=-1) if is_tensor(uv) else np.isfinite(uv).all(axis=-1)
    comparison = (uv >= margin) & (uv <= wh - 1 - margin)
    return finite & (comparison.all(dim=-1) if is_tensor(uv) else comparison.all(axis=-1))
