#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset, DataLoader
from typing import Callable, Tuple, Union
from pathlib import Path
from numpy import array
import numpy as np
from utils import (
    IPipeline,
    normalize,
    load_img,
    bgr_to_rgb,
    rgb_to_gray,
    resize
)
import os


class BasePipeline(IPipeline):
    """Pass the input into a pipeline of functions
    where each function's input depends on the last's output
    """
    def __init__(self, target_size: Tuple[int, int], *args, **kwargs) -> None:
        super().__init__()
        self.target_size = target_size


class ColoredPipeline(BasePipeline):
    """Pass the colored input image into a pipeline of functions
    where each function's input depends on the last's output
    """
    def __init__(self, target_size: Tuple[int, int], *args, **kwargs) -> None:
        super().__init__(target_size, *args, **kwargs)

    def run(self, img_path: Union[str, Path], *args, **kwargs) -> array:
        img = load_img(img_path, *args, **kwargs)
        img = bgr_to_rgb(img, *args, **kwargs)
        img = resize(img, self.target_size, *args, **kwargs)
        img = normalize(img, *args, **kwargs)
        return np.swapaxes(img, 0, -1)


class GrayImgPipeline(BasePipeline):
    """Pass the gray input image into a pipeline of functions
    where each function's input depends on the last's output
    """
    def __init__(self, target_size: Tuple[int, int], *args, **kwargs) -> None:
        super().__init__(target_size, *args, **kwargs)

    def run(self, img_path: Union[str, Path], *args, **kwargs) -> array:
        img = load_img(img_path, *args, **kwargs)
        img = bgr_to_rgb(img, *args, **kwargs)
        img = resize(img, self.target_size, *args, **kwargs)
        img = rgb_to_gray(img, *args, **kwargs)
        img = normalize(img, *args, **kwargs)
        return np.expand_dims(img, axis=0)


class ColoredData(Dataset):
    def __init__(
            self,
            data_path: Union[str, Path],
            pipeline: IPipeline,
            files_filter: Union[Callable, None] = None
            ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.data_path = data_path
        self.files = [
            os.path.join(data_path, file)
            for file in os.listdir(self.data_path)
            ]
        if files_filter is not None:
            self.files = list(filter(files_filter, self.files))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> array:
        file = self.files[index]
        return self.pipeline.run(file)


class GrayData(ColoredData):
    def __init__(
            self,
            data_path: Union[str, Path],
            pipeline: IPipeline,
            gray_pipeline: IPipeline,
            files_filter: Union[Callable, None] = None
            ) -> None:
        super().__init__(data_path, pipeline, files_filter)
        self.gray_pipeline = gray_pipeline

    def __getitem__(self, index: int) -> Tuple[array, array]:
        file = self.files[index]
        return (
            self.pipeline.run(file),
            self.gray_pipeline.run(file)
            )


def get_colored_data_loader(
        data_path: Union[str, Path],
        pipeline: IPipeline,
        files_filter: Union[Callable, None],
        batch_size: int,
        *args,
        **kwargs
        ) -> DataLoader:
    colored_dataset = ColoredData(
        data_path=data_path,
        pipeline=pipeline,
        files_filter=files_filter,
        *args,
        **kwargs
    )
    return DataLoader(
        colored_dataset,
        batch_size,
        shuffle=True,
        *args,
        **kwargs
    )


def get_gray_data_loader(
        data_path: Union[str, Path],
        pipeline: IPipeline,
        gray_pipeline: IPipeline,
        files_filter: Union[Callable, None],
        batch_size: int,
        *args,
        **kwargs
        ) -> DataLoader:
    gray_dataset = GrayData(
        data_path=data_path,
        pipeline=pipeline,
        gray_pipeline=gray_pipeline,
        files_filter=files_filter,
        *args,
        **kwargs
    )
    return DataLoader(
        gray_dataset,
        batch_size,
        shuffle=True,
        *args,
        **kwargs
    )
