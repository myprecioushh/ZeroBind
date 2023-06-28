# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from GCNmodel import GCN_DTIMAML
from data import GCNMoleculeDataModule
from meta import Meta
import torch
from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import torch.multiprocessing
import datetime

torch.multiprocessing.set_sharing_strategy('file_system')
directory=["./train_roc_curve_review_v9/","./zero_roc_curve_review_v9/"]
def cli_main():
    parser = ArgumentParser()
    parser = Meta.add_model_specific_args(parser)
    args = parser.parse_args()
    pl.seed_everything(42)
    molecule_data = GCNMoleculeDataModule.from_argparse_args(args)
    if not args.test and not args.explanation:
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time)
        for dir in directory:
            if not os.path.exists(dir):
                os.mkdir(dir)
            else:
                for root, dirs, files in os.walk(dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))  # 删除文件
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
        wandb_logger = WandbLogger(name=args.project_name, project="GCN_maml")
        args.iteration=molecule_data.iterations
        molecule_model = GCN_DTIMAML(args=args)

        dirpath = args.project_name+"+checkpoint"
        checkpoint_callback = ModelCheckpoint(
            monitor='zero_auroc',
            dirpath=dirpath,
            filename='-{epoch:03d}-{zero_auroc:.4f}-{zero_loss:.4f}-',
            save_top_k=50,
            mode='max',
            save_last=True,
        )
        trainer = pl.Trainer(devices=[0],accelerator="gpu",logger=wandb_logger,max_epochs=args.total_epoch,callbacks=[checkpoint_callback]
                         ,log_every_n_steps=1)

        trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    else:
        args.iteration = 1
        molecule_model=GCN_DTIMAML.load_from_checkpoint(args.checkpoint_path,args=args)
        trainer=pl.Trainer(devices=[0],accelerator="gpu")
    #trainer.validate(molecule_model, datamodule=molecule_data)
    if args.test or args.explanation:
        trainer.test(molecule_model, datamodule=molecule_data)

    elif args.val:
        trainer.validate(molecule_model, datamodule=molecule_data)
    else:
        trainer.fit(molecule_model,datamodule=molecule_data)


if __name__ == '__main__':
    cli_main()
