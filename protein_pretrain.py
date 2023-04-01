from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from functools import partial
from graphein.protein.edges.distance import add_distance_threshold,add_peptide_bonds
import esm
import networkx as nx
import os
import torch
import pandas
import warnings
from pandas.errors import SettingWithCopyWarning
from tqdm import tqdm
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
protein_model.eval()
new_edge_funcs = {"edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=0, threshold=8)]}
config = ProteinGraphConfig(**new_edge_funcs)

def pretrain_protein(data):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    results = protein_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    feat=token_representations.squeeze(0)[1:len(data[0][1])+1]

    return feat
def graph_node(pdb_ID,seq):
    if len(seq)>1022:
        seq_feat=[]
        for i in range(len(seq)//1022):
            data=[(pdb_ID,seq[i*1022:(i+1)*1022])]
            seq_feat.append(pretrain_protein(data))
        data=[(pdb_ID,seq[(i+1)*1022:])]
        seq_feat.append(pretrain_protein(data))
        seq_feat=torch.cat(seq_feat,dim=0)
    else:
        data=[(pdb_ID,seq)]
        seq_feat=pretrain_protein(data)

    return seq_feat
def adj2table(adj):
    edge_index=[[],[]]
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if int(adj[i][j])!=0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.tensor(edge_index,dtype=torch.long)
def protein_graph(protein_path,pdb_ID):
    g = construct_graph(config=config, pdb_path=protein_path+pdb_ID+".pdb")
    A = nx.to_numpy_array(g,nonedge=0,weight='distance')
    edge_index=adj2table(A)
    seq=""
    for key in g.graph.keys():
        if key[:9]=="sequence_":
            seq+=g.graph[key]
    if len(seq)!=g.number_of_nodes():
        raise RuntimeError("number of nodes mismatch")
    node_feat=graph_node(pdb_ID,seq)

    return edge_index,node_feat.detach()
def covid19():
    protein_graphs = {}
    covid_path = os.getcwd() + "/covid 19/"
    covid_data = pandas.read_csv(covid_path + "./covid19.csv")
    covid_protein_ID = set(covid_data["covid19_pdb"].values.tolist())
    for protein_ID in covid_protein_ID:
        protein_graphs[protein_ID] = protein_graph(covid_path,protein_ID)
    return protein_graphs

def trainandtest():
    protein_graphs = {}
    protein_path = os.getcwd() + "/tmp/"
    train_data = pandas.read_csv("./protein_train.csv")
    train_protein_ID = set(train_data["protein_ID"].values.tolist())
    val_data = pandas.read_csv("./protein_seen_node.csv")
    val_protein_ID = set(val_data["protein_ID"].values.tolist())
    test_data = pandas.read_csv("./protein_inductive.csv")
    test_protein_ID = set(test_data["protein_ID"].values.tolist())
    for protein_ID in tqdm(set.union(train_protein_ID,val_protein_ID)):
        protein_graphs[protein_ID] = protein_graph(protein_path, protein_ID)
    return protein_graphs

def pretrain_init(protein_ID):

    return explan_graphs[protein_ID]

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
#protein_graphs=covid19()
#protein_graphs=trainandtest()
explan_graphs=pre_loading()

