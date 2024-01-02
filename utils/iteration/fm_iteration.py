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
from ..loss import *
from ..metric import *
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


def fm_iteration(
        model, 
        dataloader, 
        epoch, 
        is_train, 
        device, 
        metric, 
        optimizer=None, 
        scheduler=None, 
        use_wandb=False
        ):
    type_str = 'Train' if is_train else 'Valid'

    epoch_iterator = tqdm(dataloader)
    losses = 0.0
    for iter, (y_true, fields) in enumerate(epoch_iterator, start=1):
        logits = model(fields.to(device))
        
        # Compute running criterion
        y_true = torch.Tensor(y_true.view(-1))
        y_score = logits.view(-1).cpu().detach()
        y_pred = (logits.view(-1).cpu().detach() >= 0.5).type(torch.int)
        metric.update(y_true, y_pred, y_score)
        running_accuracy = float(torch.sum((y_true==y_pred).type(torch.int))) / len(y_true)

        # Compute loss
        loss = nn.BCEWithLogitsLoss()(logits, y_true.to(device))
        losses += loss.item()
        if is_train == True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
        
        # Log
        epoch_iterator.set_description(
            f'{type_str} | Epoch: {epoch + 1:03} | Loss: {loss:.5f} | Acc: {running_accuracy:.5f}'
            )
        if use_wandb:
            log = {
                f'{type_str}_loss': loss,
                f'{type_str}_accuracy': running_accuracy,
                f'{type_str}_step': epoch * len(epoch_iterator) + iter
                }
            if is_train == True:
                log["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log)

    # Final Log
    loss = losses / iter
    acc = metric.calc_acc()
    criterion = acc
    metric.clean()
    print( f'END Epoch: {epoch + 1:03} | loss: {loss:.5f} | Acc: {acc:.5f}')
    return loss if is_train else criterion