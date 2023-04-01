import copy
from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import random
from functools import partial
from pre_process import train_molecules,test_molecules,test_explan
from torch_geometric.data import DataLoader as GNN_DataLoader
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data
import math

class FewShotBatchSampler:
    def __init__(self,train_molecule_datasets,K_shot,K_query,batch_size):

        self.molecule_labels = train_molecule_datasets.molecule_labels
        self.molecule_ID=train_molecule_datasets.molecule_ID
        self.N_way = 2
        self.K_shot = K_shot
        self.batch_size=batch_size
        self.K_query=K_query

        self.classes=[0,1]
        self.protein_name=list(self.molecule_ID.keys())
        self.iterations=0
        molecule_ID=copy.deepcopy(self.molecule_ID)
        proteins = list(molecule_ID.keys())
        while proteins:
            selected_proteins = proteins[:self.batch_size]
            for selected_protein in selected_proteins:
                positive_list = []
                negative_list = []
                for index in molecule_ID[selected_protein]:
                    if self.molecule_labels[index] == 1:
                        positive_list.append(index)
                    else:
                        negative_list.append(index)

                selected_positive = random.sample(positive_list, self.K_shot)
                selected_negative = random.sample(negative_list, self.K_shot)
                for positive_index in selected_positive:
                    molecule_ID[selected_protein].remove(positive_index)
                    positive_list.remove(positive_index)
                for negative_index in selected_negative:
                    molecule_ID[selected_protein].remove(negative_index)
                    negative_list.remove(negative_index)
                '''if len(positive_list) + len(negative_list) >= self.K_query:
                    selected_query_positive = random.sample(positive_list, self.K_query - math.ceil(self.K_query * len(negative_list) / (len(positive_list) + len(negative_list))))
                    selected_query_negative = random.sample(negative_list, math.ceil(self.K_query * len(negative_list) / (len(positive_list) + len(negative_list))))
                else:
                    selected_query_positive = copy.deepcopy(positive_list)
                    selected_query_negative = copy.deepcopy(negative_list)'''
                if len(positive_list)>=self.K_query//2 and len(negative_list)>=self.K_query//2:
                    selected_query_positive = random.sample(positive_list,self.K_query//2)
                    selected_query_negative = random.sample(negative_list,self.K_query//2)
                else:
                    length=min(len(positive_list),len(negative_list))
                    selected_query_positive=random.sample(positive_list,length)
                    selected_query_negative=random.sample(negative_list,length)
                for positive_index in selected_query_positive:
                    molecule_ID[selected_protein].remove(positive_index)
                    positive_list.remove(positive_index)
                for negative_index in selected_query_negative:
                    molecule_ID[selected_protein].remove(negative_index)
                    negative_list.remove(negative_index)
                if len(positive_list) <= self.K_shot or len(negative_list) <= self.K_shot:
                    molecule_ID.pop(selected_protein)
                    proteins.remove(selected_protein)
            self.iterations+=1
    def weighted_sample(self,molecule_ID):
        select_proteins=[]
        for _ in range(self.batch_size):
            weight=[len(value) for value in molecule_ID.values()]
            select_protein=random.choices(list(molecule_ID.keys()),weights=weight)
            molecule_ID.pop(select_protein[0])
            select_proteins.append(select_protein[0])
        return select_proteins
    def __iter__(self):
        molecule_ID=copy.deepcopy(self.molecule_ID)
        proteins=list(molecule_ID.keys())
        random.shuffle(proteins)
        for _ in range(self.iterations):
            index_batch=[]
            selected_proteins = proteins[:self.batch_size]
            for selected_protein in selected_proteins:
                positive_list=[]
                negative_list=[]
                for index in molecule_ID[selected_protein]:
                    if self.molecule_labels[index]==1:
                        positive_list.append(index)
                    else:
                        negative_list.append(index)
                selected_positive=random.sample(positive_list, self.K_shot)
                selected_negative=random.sample(negative_list, self.K_shot)
                for positive_index in selected_positive:
                    molecule_ID[selected_protein].remove(positive_index)
                    positive_list.remove(positive_index)
                for negative_index in selected_negative:
                    molecule_ID[selected_protein].remove(negative_index)
                    negative_list.remove(negative_index)
                '''if len(positive_list) + len(negative_list) >= self.K_query:
                    selected_query_positive = random.sample(positive_list, self.K_query - math.ceil(
                        self.K_query * len(negative_list) / (len(positive_list) + len(negative_list))))
                    selected_query_negative = random.sample(negative_list, math.ceil(
                        self.K_query * len(negative_list) / (len(positive_list) + len(negative_list))))
                else:
                    selected_query_positive = copy.deepcopy(positive_list)
                    selected_query_negative = copy.deepcopy(negative_list)'''
                if len(positive_list)>=self.K_query//2 and len(negative_list)>=self.K_query//2:
                    selected_query_positive = random.sample(positive_list,self.K_query//2)
                    selected_query_negative = random.sample(negative_list,self.K_query//2)
                else:
                    length=min(len(positive_list),len(negative_list))
                    selected_query_positive=random.sample(positive_list,length)
                    selected_query_negative=random.sample(negative_list,length)
                for positive_index in selected_query_positive:
                    molecule_ID[selected_protein].remove(positive_index)
                    positive_list.remove(positive_index)
                for negative_index in selected_query_negative:
                    molecule_ID[selected_protein].remove(negative_index)
                    negative_list.remove(negative_index)
                if len(positive_list)<=self.K_shot or len(negative_list)<=self.K_shot:
                    molecule_ID.pop(selected_protein)
                    proteins.remove(selected_protein)
                index_batch.extend(selected_positive)
                index_batch.extend(selected_negative)
                selected_query = selected_query_positive + selected_query_negative
                random.shuffle(selected_query)
                index_batch.extend(selected_query)
            random.shuffle(proteins)
            yield index_batch
    def __len__(self):
        return self.iterations


class Test_valBatchSampler:
    def __init__(self, Test_val_molecule_datasets,val_shot):

        self.molecule_labels=Test_val_molecule_datasets.molecule_labels
        self.molecule_ID=Test_val_molecule_datasets.molecule_ID

        self.val_shot=val_shot
        self.iterations=0
        for protein_name in list(self.molecule_ID.keys()):
            if len(self.molecule_ID[protein_name]) % self.val_shot == 0:
                self.iterations+=len(self.molecule_ID[protein_name])//self.val_shot
            else:
                self.iterations += len(self.molecule_ID[protein_name]) // self.val_shot + 1

    def __iter__(self):
        for protein_name in list(self.molecule_ID.keys()):
            if len(self.molecule_ID[protein_name]) % self.val_shot==0:
                n=len(self.molecule_ID[protein_name]) // self.val_shot
            else:
                n=len(self.molecule_ID[protein_name]) // self.val_shot+1
            for i in range(n):
                yield self.molecule_ID[protein_name][i*self.val_shot:(i+1)*self.val_shot]

    def __len__(self):
        return self.iterations

def collate(items,K_shot,K_query):
    node_feats,edge_indexs,edge_attrs,ys,protein_pdbIDs,edge_nums=[],[],[],[],[],[]
    batch_size=len(items)//(2*K_shot+K_query)
    for item in items:
        node_feats.append(item.x)
        edge_indexs.append(item.edge_index)
        edge_attrs.append(item.edge_attr)
        ys.append(item.y)
        protein_pdbIDs.append(item.protein_pdbID)
        edge_nums.append(item.edge_num)
    support_datas=[]
    proteins=[]
    for m in range(batch_size):
        support_data=[]
        for n_p in range(2*K_shot):
            ID=n_p+m*(2*K_shot+K_query)
            data=Data(x=node_feats[ID],edge_index=edge_indexs[ID],
                      edge_attr=edge_attrs[ID],y=ys[ID])
            support_data.append(data)
        support_datas.append(support_data)
        proteins.append(protein_pdbIDs[m*(2*K_shot+K_query)])
    query_datas=[]
    for l in range(batch_size):
        query_data=[]
        for ll in range(K_query):
            ID=ll+l*(2*K_shot+K_query)+2*K_shot
            data=Data(x=node_feats[ID],edge_index=edge_indexs[ID],
                      edge_attr=edge_attrs[ID],y=ys[ID])
            query_data.append(data)
        query_datas.append(query_data)
    return support_datas,query_datas,proteins

class GCNMoleculeDataModule(LightningDataModule):
    def __init__(self,num_workers,batch_size,k_shot,k_query,val_shot,test,explanation):
        super(GCNMoleculeDataModule, self).__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.val_shot = val_shot
        self.k_shot=k_shot
        self.k_query=k_query
        self.test=test
        self.explanation=explanation
        if self.explanation:
            self.explan_molecule=test_explan()
            self.explan_batch = Test_valBatchSampler(self.explan_molecule, self.val_shot)
        elif self.test:
            self.test_molecule=test_molecules()
            #self.test_molecule = test_covid19()
            self.test_batch=Test_valBatchSampler(self.test_molecule,self.val_shot)
        else:
            self.train_molecule,self.val_molecule=train_molecules()
            self.val_batch = Test_valBatchSampler(self.val_molecule, self.val_shot)
            self.train_batch=FewShotBatchSampler(self.train_molecule,batch_size=self.batch_size,K_shot=self.k_shot,K_query=self.k_query)
            self.iterations=self.train_batch.iterations
    def train_dataloader(self):
        loader = GNN_DataLoader(
            dataset=self.train_molecule,
            batch_sampler=self.train_batch,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return loader

    def val_dataloader(self):
        loader = GNN_DataLoader(
            dataset=self.val_molecule,
            batch_sampler=self.val_batch,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return loader

    def test_dataloader(self):
        loader = GNN_DataLoader(
            dataset=self.explan_molecule,
            batch_sampler=self.explan_batch,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return loader

