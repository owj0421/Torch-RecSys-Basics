import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
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
    total_loss = 0.0
    for iter, (targets, fields) in enumerate(epoch_iterator, start=1):
        # Forward
        logits = model(fields.to(device))

        # Compute running loss
        running_loss = nn.BCEWithLogitsLoss()(logits, targets.to(device))
        total_loss += running_loss.item()
        if is_train == True:
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        # Compute running criterion
        preds = logits.unsqueeze(1).cpu().detach()
        metric.update(targets, preds)
        running_metric = metric.running_compute(targets, preds)

        # Log
        epoch_iterator.set_description(
            f'[{type_str}] Epoch: {epoch + 1:03} | Loss: {running_loss:.5f} ' + running_metric.get_description()
            )
        if use_wandb:
            log = {
                f'{type_str}_loss': running_loss, 
                f'{type_str}_step': epoch * len(epoch_iterator) + iter
                }
            log.update(running_metric.get_wandb_log(type_str))
            if is_train == True:
                log["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log)

    # Final Log
    total_loss = total_loss / iter
    final_metric = metric.compute()
    criterion = final_metric.criterion
    metric.clean()
    if is_train == False:
        print( f'[E N D] Epoch: {epoch + 1:03} | loss: {total_loss:.5f} ' + final_metric.get_description() + '\n')

    return total_loss if is_train else criterion