import csv
from urllib.request import urlopen
import time
import numpy as np
import random
import copy
import math
from torch_geometric.data import Data
from features import atom_to_feature_vector,bond_to_feature_vector
from rdkit import Chem
import torch


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x
def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []

    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)


    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    x = convert_to_single_emb(torch.tensor(x))
    graph = dict()
    graph['edge_index'] = torch.tensor(edge_index)
    graph['edge_feat'] = torch.tensor(edge_attr)
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph
def train_molecules():

    np.random.seed(42)
    molecule_graphs={}
    molecule_labels={}
    with open("./protein_train.csv",'r')as f:
        reader=csv.reader(f)
        next(reader)
        for m,row in enumerate(reader):
            if row[0] not in molecule_graphs.keys():
                molecule_graphs[row[0]]=[]
                molecule_labels[row[0]]=[]
            if float(row[4])<1000:
                molecule_labels[row[0]].append(1)
            elif float(row[4])>=100000:
                molecule_labels[row[0]].append(0)
            else:
                continue
            try:
                graph = "Aaa"
            except:
                continue

            molecule_graphs[row[0]].append(graph)

        train_molecule_graphs=[]
        train_molecule_labels=[]
        train_names=[]
        train_molecule_ID={}
        start_index=0
        for train_name in molecule_graphs.keys():
            train_molecule_ID[train_name]=[i for i in range(start_index,start_index+len(molecule_graphs[train_name]))]
            start_index=start_index+len(molecule_graphs[train_name])
            train_names.extend([train_name for _ in range(len(molecule_graphs[train_name]))])
            train_molecule_graphs.extend(molecule_graphs[train_name])
            train_molecule_labels.extend(molecule_labels[train_name])

        train_set=MoleculeDataset(train_molecule_graphs,train_molecule_labels,train_molecule_ID,train_names)
    f.close()
    return train_set

class MoleculeDataset:
    def __init__(self,molecule_graphs,molecule_labels,molecule_ID,names,smiles=None):
        """
        Inputs:
            imgs - Numpy array of shape [N,32,32,3] containing all images.
            targets - PyTorch array of shape [N] containing all labels.
            img_transform - A torchvision transformation that should be applied
                            to the images before returning. If none, no transformation
                            is applied.
        """
        super().__init__()

        self.molecule_graphs = molecule_graphs
        self.molecule_labels = molecule_labels
        self.molecule_ID=molecule_ID
        self.names=names
        self.smiles=smiles
        self.data=[]


    def __getitem__(self,idx):
        return self.names[idx]

    def __len__(self):
        return len(self.molecule_graphs)

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
                if len(positive_list) + len(negative_list) >= self.K_query:
                    selected_query_positive = random.sample(positive_list, self.K_query - math.ceil(self.K_query * len(negative_list) / (len(positive_list) + len(negative_list))))
                    selected_query_negative = random.sample(negative_list, math.ceil(self.K_query * len(negative_list) / (len(positive_list) + len(negative_list))))
                else:
                    selected_query_positive = copy.deepcopy(positive_list)
                    selected_query_negative = copy.deepcopy(negative_list)
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
                if len(positive_list) + len(negative_list) >= self.K_query:
                    selected_query_positive = random.sample(positive_list, self.K_query - math.ceil(
                        self.K_query * len(negative_list) / (len(positive_list) + len(negative_list))))
                    selected_query_negative = random.sample(negative_list, math.ceil(
                        self.K_query * len(negative_list) / (len(positive_list) + len(negative_list))))
                else:
                    selected_query_positive = copy.deepcopy(positive_list)
                    selected_query_negative = copy.deepcopy(negative_list)
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

if __name__=="__main__":
    train_set=train_molecules()
    test=FewShotBatchSampler(train_set,batch_size=4,K_shot=5,K_query=50)
    for id in test:
        print(id)












