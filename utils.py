#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Tuple, Union
from datetime import datetime
from torch.nn import Module
from pathlib import Path
from numpy import array
import torch
import cv2


class IPipeline(ABC):
    @abstractmethod
    def run():
        """Used to run all the callables functions sequantially
        """
        pass


def normalize(img: array) -> array:
    """Used to normalize the image's pixels to become between 0 and 1

    Args:
        img (array): The image to be normalized

    Returns:
        array: normalized image
    """
    return img / 255.0


def load_img(img_path: Union[Path, str]) -> array:
    """Used to load an image from the given path

    Args:
        img_path (Union[Path, str]): the path of the image to be loadded

    Returns:
        array: an array of the image
    """
    return cv2.imread(img_path)


def bgr_to_rgb(img: array) -> array:
    """Used swap the red and blue color channels

    Args:
        img (array): array of the BGR image

    Returns:
        array: array of RGB image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_gray(img: array) -> array:
    """Used to convert RGB image to gray scale image

    Args:
        img (array): array of RGB image

    Returns:
        array: array of gray scale image
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def resize(img: array, target_size: Tuple[int, int]) -> array:
    """Used to resize the input image to the target shape

    Args:
        img (array): the image to be resized
        target_size (Tuple[int, int]): the target shape

    Returns:
        array: the resized image
    """
    if img.shape[:2] != target_size:
        return cv2.resize(img, target_size)
    return img


def get_formated_date() -> str:
    """Used to generate time stamp

    Returns:
        str: a formated string represnt the current time stap
    """
    t = datetime.now()
    return f'{t.year}{t.month}{t.day}-{t.hour}{t.minute}{t.second}'


def load_stat_dict(model: Module, model_path: Union[str, Path]) -> None:
    """Used to load the weigths for the given model

    Args:
        model (Module): the model to load the weights into
        model_path (Union[str, Path]): tha path of the saved weigths
    """
    model.load_state_dict(torch.load(model_path))
