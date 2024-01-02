import os
import wandb
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from datetime import datetime
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from .loss import *
from .metric import *
from .iteration.fm_iteration import fm_iteration
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class TrainingArguments:
    model_name: str = 'FM'
    train_batch: int = 8
    valid_batch: int = 32
    n_epochs: int = 100
    learning_rate: float = 0.01
    save_every: int = 1
    work_dir: str = None
    use_wandb: bool = False
    device: str = 'cuda'


class Trainer:
    
    def __init__(
            self,
            args: TrainingArguments,
            model: nn.Module,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            metric: MetricCalculator,
            scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None
            ):
        self.device = torch.device(args.device)

        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.scheduler = scheduler
        self.metric = metric
        self.args = args

        self.best_state = {}

    def fit(self):
        best_criterion = -np.inf
        for epoch in range(self.args.n_epochs):
            loss = self._train(epoch, self.train_dataloader)
            criterion = self._validate(epoch, self.valid_dataloader)
            if criterion > best_criterion:
               best_criterion = criterion
               self.best_state['model'] = deepcopy(self.model.state_dict())

            if epoch % self.args.save_every == 0:
                date = datetime.now().strftime('%Y-%m-%d')
                output_dir = os.path.join(self.args.work_dir, 'checkpoints', self.args.model_name, date)
                model_name = f'{epoch}_{best_criterion:.3f}'
                self._save(output_dir, model_name)


    def _train(self, epoch: int, dataloader: DataLoader):
        self.model.train()
        is_train=True
        if self.args.model_name == 'fm':
            loss = fm_iteration(self.model, dataloader, epoch, is_train, self.device, self.metric, self.optimizer, self.scheduler, self.args.use_wandb)
        return loss


    @torch.no_grad()
    def _validate(self, epoch: int, dataloader: DataLoader):
        self.model.eval()
        is_train=False
        if self.args.model_name == 'fm':
            criterion = fm_iteration(self.model, dataloader, epoch, is_train, self.device, self.metric, self.optimizer, self.scheduler, self.args.use_wandb)
        return criterion


    def _save(self, dir, model_name, best_model: bool=True):
        def _create_folder(dir):
            try:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            except OSError:
                print('[Error] Creating directory.' + dir)
        _create_folder(dir)

        path = os.path.join(dir, f'{model_name}.pth')
        checkpoint = {
            'model_state_dict': self.best_state['model'] if best_model else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
            }
        torch.save(checkpoint, path)
        print(f'[COMPLETE] Save at {path}')


    def load(self, path, load_optim=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=False)
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'], strict=False)
        print(f'[COMPLETE] Load from {path}')