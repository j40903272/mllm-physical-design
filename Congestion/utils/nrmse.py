# Copyright 2022 CircuitNet. All rights reserved.

from functools import wraps
from inspect import getfullargspec

import os
import os.path as osp
import cv2
import numpy as np
import torch
import multiprocessing as mul
import uuid
import psutil
import time
import csv
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from scipy.interpolate import make_interp_spline
from functools import partial
from mmcv import scandir

from scipy.stats import wasserstein_distance
from skimage.metrics import normalized_root_mse
import math


def input_converter(apply_to=None):
    def input_converter_wrapper(old_func):
        @wraps(old_func)
        def new_func(*args, **kwargs):
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(tensor2img(args[i]))
                    else:
                        new_args.append(args[i])

            return old_func(*new_args)
        return new_func

    return input_converter_wrapper


@input_converter(apply_to=('img1', 'img2'))
def nrms(img1, img2, crop_border=0):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    nrmse_value = normalized_root_mse(img1.flatten(), img2.flatten(),normalization='min-max')
    if math.isinf(nrmse_value):
        return 0.05
    return nrmse_value


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).squeeze(0)
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()

        if n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[:, :, :], (2, 0, 1))
            # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 2:
            img_np = _tensor.numpy()[..., None]
        else:
            raise ValueError('Only support 4D, 3D or 2D tensor. '
                             f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result