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
from protein_pretrain import protein_graph
from tqdm import tqdm

def pre_loading():
    protein_graphs = {}
    with open("./INDEX_general_PL.2020","r")as f:
        PDB_IDs, affinitys=[],[]
        for i,line in enumerate(f.readlines()):
            if i>=6:
                data=line.split(" ")
                PDB_ID,affinity=data[0],data[3]
                try:
                    affinity=affinity[affinity.index("="):]
                except:
                    continue
                PDB_IDs.append(PDB_ID)
                affinitys.append(affinity)
    for protein_ID in tqdm(PDB_IDs):
        protein_graphs[protein_ID] = protein_graph("/amax/yxwang/GCN/v2020-other-PL/", protein_ID+"/"+protein_ID+"_protein")
    return protein_graphs


