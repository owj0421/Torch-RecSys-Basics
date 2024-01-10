import os
import wandb
import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch_recsys_implementations.model.fm.factorization_machines import *
from model.fm.deep_fm import *
from model.fm.wide_and_deep import *
from data.builder import *
from utils.trainer import *

import sys
sys.path.append('F:/Projects')
from torch_recsys_metrics.src.torch_recsys_metrics.calculator import * 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Parser
parser = argparse.ArgumentParser(description='Outfit-Transformer Trainer')
parser.add_argument('--model_name', help='~', type=str, default='wide_and_deep')
parser.add_argument('--train_batch', help='Size of Batch for Training', type=int, default=1024)
parser.add_argument('--valid_batch', help='Size of Batch for Validation, Test', type=int, default=1024)
parser.add_argument('--n_epochs', help='Number of epochs', type=int, default=20)
parser.add_argument('--scheduler_step_size', help='Step LR', type=int, default=1000)
parser.add_argument('--save_every', help='Learning rate', type=int, default=100)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=1e-4)
parser.add_argument('--work_dir', help='Full working directory', type=str, default='F:/Projects/torch_recsys_implementations')
parser.add_argument('--data_dir', help='Full dataset directory', type=str, default='F:/Projects/datasets/ml-latest-small')
parser.add_argument('--wandb_api_key', default=None)#'fa37a3c4d1befcb0a7b9b4d33799c7bdbff1f81f')
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
    use_wandb = True if args.wandb_api_key else False,
    save_every=args.save_every,
    )

builder_output = build_dataset(
    args.data_dir, 
    type='fm',
    threshold = 4.0,
    n_user = None, 
    n_test = 5
    )

train_dataset, valid_dataset = random_split(builder_output['train_dataset'], (0.8, 0.2))
train_dataloader = DataLoader(train_dataset, training_args.train_batch, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, training_args.valid_batch, shuffle=False)
print('[COMPLETE] Build Dataset, DataLoader')

if args.model_name == 'fm':
    model = FactorizationMachines(wide_features=builder_output['wide_features'], deep_features=builder_output['deep_features'], feature2index=builder_output['feature2index']).to(device)
elif args.model_name == 'deep_fm':
    model = DeepFM(wide_features=builder_output['wide_features'], deep_features=builder_output['deep_features'], feature2index=builder_output['feature2index']).to(device)
elif args.model_name == 'wide_and_deep':
    model = WideandDeep(wide_features=builder_output['wide_features'], deep_features=builder_output['deep_features'], feature2index=builder_output['feature2index']).to(device)
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