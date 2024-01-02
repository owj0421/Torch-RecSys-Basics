import os
import wandb
import argparse
from itertools import chain

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.factorization_machine.fm import *
from utils.trainer import *
from utils.dataset import *
from utils.metric import MetricCalculator

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Parser
parser = argparse.ArgumentParser(description='Outfit-Transformer Trainer')
parser.add_argument('--model_name', help='~', type=str, default='fm')
parser.add_argument('--train_batch', help='Size of Batch for Training', type=int, default=512)
parser.add_argument('--valid_batch', help='Size of Batch for Validation, Test', type=int, default=512)
parser.add_argument('--n_epochs', help='Number of epochs', type=int, default=5)
parser.add_argument('--scheduler_step_size', help='Step LR', type=int, default=1000)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=1e-2)
parser.add_argument('--work_dir', help='Full working directory', type=str, default='F:/Projects/torch-recsys-Implementations')
parser.add_argument('--data_dir', help='Full dataset directory', type=str, default='F:/Projects/datasets/ml-latest-small')
parser.add_argument('--wandb_api_key', default=None)
parser.add_argument('--checkpoint', default=None)
args = parser.parse_args()

# Wandb
if args.wandb_api_key:
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_PROJECT"] = f"torch-recsys-{args.model_name}"
    os.environ["WANDB_LOG_MODEL"] = "all"
    wandb.login()
    run = wandb.init()

# Setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

training_args = TrainingArguments(
    model_name=args.model_name,
    train_batch=args.train_batch,
    valid_batch=args.valid_batch,
    n_epochs=args.n_epochs,
    learning_rate=args.learning_rate,
    work_dir = args.work_dir,
    use_wandb = True if args.wandb_api_key else False
    )

movie_data, dataset, test_dataset = get_dataset(
    args.data_dir, 
    threshold = 4.0,
    n_user = None, 
    n_test = 5
    )

train_dataset, valid_dataset = random_split(dataset, (0.8, 0.2))
train_dataloader = DataLoader(train_dataset, training_args.train_batch, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, training_args.valid_batch, shuffle=False)
print('[COMPLETE] Build Dataset, DataLoader')

if args.model_name == 'fm':
    model = FM(field_dims=dataset.field_dims, embed_dim=32).to(device)
else:
    raise ValueError('')

print(f'[COMPLETE] Build {args.model_name} Model')

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.5)
metric = MetricCalculator()
trainer = Trainer(training_args, model, train_dataloader, valid_dataloader,
                  optimizer=optimizer, metric=metric , scheduler=scheduler)

if args.checkpoint != None:
    checkpoint = args.checkpoint
    trainer.load(checkpoint, load_optim=False)
    print(f'[COMPLETE] Load Model from {checkpoint}')

# Train
trainer.fit()