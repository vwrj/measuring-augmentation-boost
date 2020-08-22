import pdb
import argparse
import numpy as np
import sklearn
from collections import OrderedDict
from test_tube import Experiment
from dataset import CIFAR10DataModule

import torch
import torch.nn as nn
from torchvision import transforms, utils
from torchvision.models import resnet18
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


class LossTracker():
    def __init__(self, *items_to_track):
        '''
        items_to_track : Collects all losses to track into a tuple of strings.
            for example ('global_loss', 'local_loss')

        '''
        self.l = dict()
        for x in items_to_track:
            self.l[x] = []

    def add(self, name, value):
        self.l[name].append(value)

    def get_mean(self, name):
        assert name in self.l.keys()
        return np.mean(self.l[name])

    def get_acc(self, name):
        assert name in ['train_acc', 'val_acc']
        return sum(b[0] for b in self.l[name]) * 100 / sum(b[1] for b in self.l[name])


    def reset(self):
        for x in self.l:
            self.l[x] = []


class CIFAR10Model(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters() 

        self.data_module = data_module

        self.model = resnet18()
        self.model.fc = nn.Linear(512, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.lt = LossTracker('train_loss', 'val_loss', 'train_acc', 'val_acc')

    def forward(self, batch):
        imgs, y, _ = batch
        y_hat = self.model(imgs)
        loss = self.criterion(y_hat, y)

        predictions = torch.argmax(output, 1)
        correct = torch.sum(predictions == y).item()
        return loss, correct

    def training_step(self, batch, batch_idx):
        loss, correct = self.forward(batch)

        self.lt.add('train_loss', loss.item())
        self.lt.add('train_acc', (correct.item(), batch.shape[0]))
        logger_logs = {'avg_train_loss': self.lt.get_mean('train_loss'), 
                       'running_train_acc': self.lt.get_accuracy('train_acc')}
        output = {
            'loss': loss,
            'accuracy': correct.item() / batch.shape[0],
            'log': logger_logs
        }

        return output

    def validation_step(self, batch, batch_idx):
        loss, correct = self.forward(batch)

        self.lt.add('val_loss', loss.item())
        self.lt.add('val_acc', (correct.item(), batch.shape[0]))
        tb_logs = {'avg_val_loss': self.lt.get_mean('val_loss'),
                   'running_val_acc': self.lt.get_accuracy('val_acc')}
        output = {
            'val_loss': loss,
            'val_accuracy': correct.item() / batch.shape[0],
            'log': tb_logs
        }

        return output

    def validation_epoch_end(self, outputs):

        # Add Accuracy Tracker
        avg_val_loss = self.lt.get_mean('val_loss')
        val_acc = self.lt.get_accuracy('val_acc')
        tb_logs = {'val_loss': avg_val_loss,
                   'val_accuracy': val_acc}
        self.lt.reset()

        return {'val_loss': avg_val_loss, 'log': tb_logs}

    def train_dataloader(self):
        return self.data_module.train_dataloader(
                augment = self.hparams.augment, 
                size = self.hparams.size,
                bs = self.hparams.batch_size,
        )


    def val_dataloader(self):
        return self.data_module.val_dataloader(
                size = self.hparams.size,
                bs = self.hparams.batch_size
        )

    def configure_optimizers(self):
        optimizers = [
                torch.optim.SGD(self.parameters(),
                    lr = self.hparams.learning_rate,
                    momentum=0.9, weight_decay = self.hparams.l2_penalty
            )]
        return optimizers 


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--logs-dir', required=True)
    parser.add_argument('--experiment-dir', default='experiments_lightning')
    parser.add_argument('--l2-penalty', type=float, default=5e-4)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--size', type=str)

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.track_grad_norm = int(args.track_grad_norm)

    data_module = CIFAR10DataModule()
    model = CIFAR10Model(data_module, **vars(args))

    logger = TensorBoardLogger(f'{args.experiment_dir}/{args.logs_dir}', name=f'{args.logs_dir}')
    # init trainer... all flags are available via CLI (--gpus, --max_epochs)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    trainer.fit(model)
