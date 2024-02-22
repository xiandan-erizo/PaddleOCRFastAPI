# -*- coding: utf-8 -*-

import base64

import cv2
import numpy as np


def base64_to_ndarray(b64_data: str):
    """base64转numpy数组

    Args:
        b64_data (str): base64数据

    Returns:
        _type_: _description_
    """
    image_bytes = base64.b64decode(b64_data)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


def bytes_to_ndarray(img_bytes: str):
    """字节转numpy数组

    Args:
        img_bytes (str): 图片字节

    Returns:
        _type_: _description_
    """
    image_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image_np2


def convert_numpy(obj: any) -> any:
    """递归转换 NumPy 数据类型到 Python 数据类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(item) for item in obj)
    else:
        return obj
