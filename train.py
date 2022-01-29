#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import get_formated_date, load_stat_dict
from typing import Callable, Tuple, Union
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from omegaconf import OmegaConf
from functools import wraps
from torch.nn import Module
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
from model import (
    Generator,
    Discriminator,
    Loss
    )
from data import (
    ColoredPipeline,
    GrayImgPipeline,
    get_colored_data_loader,
    get_gray_data_loader
)
import os
import torch


CONFIG = OmegaConf.load('configs/config.yaml')
IMG_SIZE = (64, 64)


def save_checkpoint(func) -> Callable:
    """Save a checkpoint after each iteration
    """
    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        result = func(obj, *args, **kwargs)
        if not os.path.exists(obj.checkpoint_dir):
            os.mkdir(obj.checkpoint_dir)
        timestamp = get_formated_date()
        gen_path = os.path.join(
            obj.checkpoint_dir, 'gen_' + timestamp + '.pt'
            )
        disc_path = os.path.join(
            obj.checkpoint_dir, 'disc_' + timestamp + '.pt'
            )
        torch.save(obj.generator.state_dict(), gen_path)
        torch.save(obj.discriminator.state_dict(), disc_path)
        print(f'{timestamp} checkpoint saved')
        return result
    return wrapper


class Trainer:
    __disc_loss_key = 'disc_loss'
    __adv_loss_key = 'adv_loss'
    __test_loss_key = 'test_loss'

    def __init__(
            self,
            loss: Loss,
            gen_optimizer: Optimizer,
            disc_optimizer: Optimizer,
            generator: Module,
            discriminator: Module,
            gray_loader_1: DataLoader,
            gray_loader_2: DataLoader,
            colored_loader: DataLoader,
            device: str,
            test_loader: DataLoader,
            checkpoint_dir: Union[Path, str],
            epochs: int
            ) -> None:
        self.loss = loss
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.generator = generator
        self.discriminator = discriminator
        self.gray_loader_1 = gray_loader_1
        self.gray_loader_2 = gray_loader_2
        self.colored_loader = colored_loader
        self.device = device
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.step_history = dict()
        self.history = dict()

    def test(self):
        """Iterate over the whole test data and test the models
        for a single epoch
        """
        test_loss = 0
        self.set_test_mode()
        for y, x in tqdm(self.test_loader):
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            fake_out = self.generator(x)
            adv_out = self.discriminator(fake_out).squeeze()
            total_loss = self.loss.calc_mae(fake_out, y)
            total_loss += self.loss.calc_bce_target_1(adv_out)
            test_loss += total_loss.item()
        test_loss /= len(self.test_loader)
        if self.__test_loss_key in self.history:
            self.history[self.__test_loss_key].append(test_loss)
        else:
            self.history[self.__test_loss_key] = [test_loss]

    def fit(self):
        """The main training loop that train the model on the training
        data then test it on the test set and then log the results
        """
        for _ in range(self.epochs):
            self.train()
            self.test()
            self.log_results()
            self.print_results()
            self.reset_log()

    def set_train_mode(self) -> None:
        """Set the models on the training mood
        """
        self.generator = self.generator.train()
        self.discriminator = self.discriminator.train()

    def set_test_mode(self) -> None:
        """Set the models on the testing mood
        """
        self.generator = self.generator.eval()
        self.discriminator = self.discriminator.eval()

    def disc_step(self, real: Tensor, fake: Tensor, *args, **kwargs) -> None:
        """Trains discriminator to get better at
        classifying fake and real data

        Args:
            real (Tensor): sample from the real data distribution
            fake (Tensor): sample from the fake data
        """
        self.disc_optimizer.zero_grad()
        real_out = self.discriminator(real).squeeze()
        fake_out = self.discriminator(self.generator(fake)).squeeze()
        total_loss = self.loss.calc_bce_target_1(real_out)
        total_loss += self.loss.calc_bce_target_0(fake_out)
        total_loss.backward()
        self.disc_optimizer.step()
        if self.__disc_loss_key in self.step_history:
            self.step_history[self.__disc_loss_key].append(total_loss.item())
        else:
            self.step_history[self.__disc_loss_key] = [total_loss.item()]

    def gen_step(
            self,
            adv_fake: Tensor,
            target_adv_fake: Tensor,
            *args,
            **kwargs
            ) -> None:
        """Performs an adversarial step on discriminator,
        and to train the generator to get better at generating images

        Args:
            adv_fake (Tensor): image to be used as adversarial image, which
            will be fed into the generator
            and then the result into the discriminator
            target_adv_fake (Tensor): The expected target from the
            generator (the ground truth)
        """
        self.gen_optimizer.zero_grad()
        fake_out = self.generator(adv_fake)
        adv_out = self.discriminator(fake_out).squeeze()
        total_loss = self.loss.calc_mae(fake_out, target_adv_fake)
        total_loss += self.loss.calc_bce_target_1(adv_out)
        total_loss.backward()
        self.gen_optimizer.step()
        if self.__adv_loss_key in self.step_history:
            self.step_history[self.__adv_loss_key].append(total_loss.item())
        else:
            self.step_history[self.__adv_loss_key] = [total_loss.item()]

    def log_results(self) -> None:
        """Calculate the cumulative results over one epoch
        """
        for key, value in self.step_history.items():
            if key in self.history:
                self.history[key].append(sum(value)/len(value))
            else:
                self.history[key] = [sum(value)/len(value)]

    def reset_log(self) -> None:
        """Reset the steps' results
        """
        for key in self.step_history.keys():
            self.step_history[key] = list()

    def print_results(self):
        """Prints the results after each epoch
        """
        result = ''
        for key, value in self.history.items():
            result += f'{key}: {str(value[-1])}, '
        print(result[:-2])

    @save_checkpoint
    def train(self):
        """Iterates over the whole training data and train the models
        for a single epoch
        """
        self.set_train_mode()
        gray_iter_1 = iter(self.gray_loader_1)
        gray_iter_2 = iter(self.gray_loader_2)
        for batch in tqdm(self.colored_loader):
            _, fake = next(gray_iter_1)
            target_adv_fake, adv_fake = next(gray_iter_2)
            batch = batch.float().to(self.device)
            fake = fake.float().to(self.device)
            adv_fake = adv_fake.float().to(self.device)
            target_adv_fake = target_adv_fake.float().to(self.device)
            self.disc_step(batch, fake)
            self.gen_step(adv_fake, target_adv_fake)


def load_models() -> Tuple[Module, Module]:
    """Loads the generator and discriminator and their weights

    Returns:
        tuple: the generator and discriminator models
    """
    generator = Generator(
        **CONFIG.generator,
        device=CONFIG.device
        )
    discriminator = Discriminator(
        **CONFIG.discriminator
        ).to(CONFIG.device)
    if CONFIG.checkpoint.gen_path is not None:
        load_stat_dict(generator, CONFIG.checkpoint.gen_path)

    if CONFIG.checkpoint.disc_path is not None:
        load_stat_dict(discriminator, CONFIG.checkpoint.disc_path)
    return generator, discriminator


def files_filter(file_path: str):
    return CONFIG.data.img_ext in file_path


def get_trainer() -> Trainer:
    """Loads all the parameters and the confguration into a trainer
    object and return it back
    """
    generator, discriminator = load_models()
    loss = Loss(CONFIG.device)
    gen_opt = torch.optim.Adam(generator.parameters(), lr=1e-4)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    colored_pipeline = ColoredPipeline(IMG_SIZE)
    gray_pipeline = GrayImgPipeline(IMG_SIZE)
    gray_loader_1 = get_gray_data_loader(
        CONFIG.data.train_path,
        colored_pipeline,
        gray_pipeline,
        files_filter,
        CONFIG.tr_params.batch_size
        )
    gray_loader_2 = get_gray_data_loader(
        CONFIG.data.train_path,
        colored_pipeline,
        gray_pipeline,
        files_filter,
        CONFIG.tr_params.batch_size
        )
    colored_loader = get_colored_data_loader(
        CONFIG.data.train_path,
        colored_pipeline,
        files_filter,
        CONFIG.tr_params.batch_size
        )
    test_loader = get_gray_data_loader(
        CONFIG.data.test_path,
        colored_pipeline,
        gray_pipeline,
        files_filter,
        CONFIG.tr_params.batch_size
        )
    return Trainer(
        loss,
        gen_opt,
        disc_opt,
        generator,
        discriminator,
        gray_loader_1,
        gray_loader_2,
        colored_loader,
        CONFIG.device,
        test_loader,
        CONFIG.tr_params.save_to,
        CONFIG.tr_params.epochs
    )


if __name__ == '__main__':
    trainer = get_trainer()
    trainer.fit()
