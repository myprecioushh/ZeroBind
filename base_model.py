import torch
import math
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
from torch.nn import Linear,BatchNorm1d
from torch_geometric.nn import  GCNConv,global_mean_pool
from torch_geometric.utils import to_dense_adj,unbatch_edge_index
from torch.nn.utils import vector_to_parameters,parameters_to_vector
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('agg')


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def update_params(loss, model,update_lr):
    grads = torch.autograd.grad(loss, model.parameters())
    return parameters_to_vector(model.parameters()) - parameters_to_vector(grads) * update_lr

def plot(x,y,xlabel,ylabel,protein_name,dir):

    plt.plot(x,y,',-',color = 'g')#o-:圆形
    plt.xlabel(xlabel)#横坐标名字
    plt.ylabel(ylabel)
    plt.savefig(dir+ylabel+"_"+str(protein_name)+".png",dpi=300)
    plt.close()

class LSLRGradientDescentLearningRule(nn.Module):
    def __init__(self, device, total_num_inner_loop_steps,init_learning_rate=1e-3):

        super(LSLRGradientDescentLearningRule, self).__init__()
        print(init_learning_rate)
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate.to(device)
        self.total_num_inner_loop_steps = total_num_inner_loop_steps

    def initialise(self, names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        for idx, (key, param) in enumerate(names_weights_dict.items()):
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate,
                requires_grad=True)
    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, step_idx):

        return {
            key: names_weights_dict[key]
                 - self.names_learning_rates_dict[key.replace(".", "-")][step_idx]
                 * names_grads_wrt_params_dict[key]
            for key in names_grads_wrt_params_dict.keys()
        }

class forwardmodel(torch.nn.Module):
    def __init__(self,protein_dim1, protein_dim2,protein_dim3,molecule_dim1, molecule_dim2,hidden_dim, hidden_dim2):
        super(forwardmodel, self).__init__()
        self.molecule_atomencoder=nn.Embedding(512 * 9 + 1, molecule_dim1, padding_idx=0)
        self.protein_GCN=GCN(protein_dim1,protein_dim2,protein_dim3)
        self.molecule_GCN=moleculeGCN(molecule_dim1,molecule_dim2,hidden_dim)
        self.protein_subgraph=GIBGCN(protein_dim3,hidden_dim)

        self.cat_MLP=MLP(2*hidden_dim,hidden_dim2,1)
        self.fc1 = Linear(protein_dim3, protein_dim3)
        self.fc2 = Linear(protein_dim3, hidden_dim)
        self.protein_discriminator=Discriminator(protein_dim3)
        self.discriminator_inner_loop=20
        self.discriminator_lr=0.01

    def forward(self,protein_node_feat,protein_edge_index,node_feat,edge_index,edge_attr,batch,mode="train"):
        node_feat=self.molecule_atomencoder(node_feat)
        node_feat=torch.mean(node_feat,dim=-2)

        protein_emb=self.protein_GCN(protein_node_feat,protein_edge_index)
        molecule_embedding=self.molecule_GCN(node_feat,edge_index,edge_attr,batch)

        protein_emb=protein_emb.repeat(molecule_embedding.shape[0],1,1)
        molecule_embedding_con=molecule_embedding.unsqueeze(1).repeat(1,protein_emb.shape[1],1)
        out,protein_subgraph_emb,protein_graph_embedding,protein_pos_penalty,protein_assignment=self.protein_subgraph(protein_emb,protein_edge_index,[0 for _ in range(protein_emb.shape[0])],molecule_embedding_con)

        molecule_graph_embedding = F.relu(self.fc1(molecule_embedding))
        molecule_graph_embedding = F.dropout(molecule_graph_embedding, p=0.5, training=self.training)
        molecule_graph_embedding = self.fc2(molecule_graph_embedding)

        pred=self.cat_MLP(torch.cat([out,molecule_graph_embedding],dim=1))
        if mode=="train":
            protein_mi_loss=self.sum_loss(self.protein_discriminator,protein_graph_embedding,protein_subgraph_emb)
            return pred,protein_pos_penalty,protein_mi_loss,torch.mean(torch.cat([protein_subgraph_emb,molecule_embedding],dim=1),dim=0,keepdim=True)
        else:
            return pred,protein_assignment

    def sum_loss(self,discriminator,all_graph_embedding,subgraph_embedding):
        min_loss=float("inf")
        min_parm=parameters_to_vector(discriminator.parameters())
        for j in range(0, self.discriminator_inner_loop):
            #optimizer_local.zero_grad()
            discriminator.zero_grad()
            local_loss = -self.MI_Est(discriminator, all_graph_embedding.detach(), subgraph_embedding.detach())
            if local_loss.item()<min_loss:
                min_loss=local_loss.item()
                min_parm=parameters_to_vector(discriminator.parameters())
            #local_losses.append(-local_loss.detach().item())
            new_parm=update_params(local_loss,discriminator,self.discriminator_lr)
            vector_to_parameters(new_parm,discriminator.parameters())
        vector_to_parameters(min_parm,discriminator.parameters())
        #plot(np.arange(self.discriminator_inner_loop).tolist(),local_losses,"inner_loop","dis_loss",str(batch_idx),dir="./dis_loss/")
        mi_loss = torch.clamp(self.MI_Est(discriminator, all_graph_embedding, subgraph_embedding),0,5)

        return mi_loss
    def MI_Est(self,discriminator, graph_embedding, subgraph_embedding):

        batch_size = graph_embedding.shape[0]

        shuffle_embeddings = subgraph_embedding[torch.randperm(batch_size)]

        joint = discriminator(graph_embedding,subgraph_embedding)

        margin = discriminator(graph_embedding,shuffle_embeddings)

        mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(margin)))
        return mi_est

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lin1 = Linear(input_dim,hidden_dim)
        self.lin2 = Linear(hidden_dim,hidden_dim)
        self.lin3 = Linear(hidden_dim,output_dim)

    def forward(self,input):
        x=F.relu(self.lin1(input))
        x=F.dropout(x,training=self.training)
        x=F.relu(self.lin2(x))
        x=F.dropout(x,training=self.training)
        x=self.lin3(x)

        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_layers=4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,add_self_loops=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,add_self_loops=True))
        self.conv2 = GCNConv(hidden_channels, out_channels,add_self_loops=True)
    def forward(self, x, edge_index):
        x=self.conv1(x,edge_index)
        x = F.dropout(F.relu(x),0.5,training=self.training)
        for conv in self.convs:
            x=conv(x,edge_index)
            x = F.dropout(F.relu(x),0.5,training=self.training)
        x=self.conv2(x,edge_index)
        return x

class moleculeGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_layers=3):
        super(moleculeGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=True))
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)

        self.pool = global_mean_pool


    def forward(self, x,edge_index,edge_attr,batch):
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), 0.5, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), 0.5, training=self.training)
        x = self.conv2(x, edge_index)
        batch = torch.sub(input=batch, other=batch[0])
        x = self.pool(x, batch)

        return x
class GIBGCN(torch.nn.Module):
    def __init__(self, protein_dim3,hidden_dim):
        super(GIBGCN, self).__init__()

        self.cluster1 = Linear(protein_dim3+hidden_dim, protein_dim3)
        self.cluster=nn.ModuleList()

        self.cluster2 = Linear(protein_dim3, 2)
        self.mse_loss = nn.MSELoss()
        self.fc1=Linear(protein_dim3,protein_dim3)

        self.fc2=Linear(protein_dim3,hidden_dim)

    def assignment(self,x):
        x = F.relu(self.cluster1(x))
        x=self.cluster2(x)
        return x

    #for i in batch:
    def aggregate(self, assignment, x, batch, edge_index):
        if assignment.get_device()<0:
            batch = torch.tensor(batch, device="cpu")
            max_id = torch.max(batch)
            EYE = torch.ones(2, device="cpu")
        else:
            batch=torch.tensor(batch,device="cuda:"+str(assignment.get_device()))
            max_id = torch.max(batch)
            EYE = torch.ones(2,device="cuda:"+str(assignment.get_device()))


        all_pos_penalty = 0
        all_graph_embedding = []
        all_pos_embedding = []

        st = 0
        end = 0

        for i in range(int(max_id + 1)):

            #print(i)

            j = 0

            while batch[st + j] == i and st + j <= len(batch) - 2:
                j += 1

            end = st + j

            if end == len(batch) - 1:
                end += 1
            one_batch_x = x[st:end]
            one_batch_assignment = assignment[st:end]

            #sum_assignment = torch.sum(x, dim=0, keepdim=False)[0]
            group_features=one_batch_assignment[:,:,0].unsqueeze(-1)
            pos_embedding=group_features*one_batch_x
            pos_embedding=torch.mean(pos_embedding,dim=1)
            #group_features = torch.mm(torch.t(one_batch_assignment), one_batch_x)

            #pos_embedding = group_features[0].unsqueeze(dim=0)

            Adj = to_dense_adj(edge_index)[0]
            new_adj = torch.matmul(torch.transpose(one_batch_assignment,1,2), Adj)
            new_adj = torch.matmul(new_adj, one_batch_assignment)


            normalize_new_adj = F.normalize(new_adj, p=1, dim=2,eps = 0.00001)

            #pos_penalty=torch.linalg.matrix_norm(normalize_new_adj-EYE)
            if assignment.get_device() < 0:
                pos_penalty = torch.tensor(0.0, device="cpu")
            else:
                pos_penalty=torch.tensor(0.0,device="cuda:"+str(assignment.get_device()))
            for p in range(normalize_new_adj.shape[0]):
                norm_diag = torch.diag(normalize_new_adj[p])
                pos_penalty += self.mse_loss(norm_diag, EYE)/normalize_new_adj.shape[0]

            graph_embedding = torch.mean(one_batch_x, dim=1)

            all_pos_embedding.append(pos_embedding)
            all_graph_embedding.append(graph_embedding)

            all_pos_penalty = all_pos_penalty + pos_penalty

            st = end



        all_pos_embedding = torch.cat(tuple(all_pos_embedding), dim=0)
        all_graph_embedding = torch.cat(tuple(all_graph_embedding), dim=0)
        all_pos_penalty = all_pos_penalty / (max_id + 1)

        return all_pos_embedding,all_graph_embedding, all_pos_penalty

    def forward(self, emb, edge_index, batch,molecule_embedding):

        assignment = torch.nn.functional.softmax(self.assignment(torch.cat([emb,molecule_embedding],dim=-1)), dim=-1)
        all_subgraph_embedding, all_graph_embedding,all_pos_penalty = self.aggregate(assignment, emb, batch, edge_index)


        out = F.relu(self.fc1(all_subgraph_embedding))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)
        return out,all_subgraph_embedding, all_graph_embedding, all_pos_penalty,assignment


class Discriminator(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()

        self.input_size = 2 * hidden_size
        self.hidden_size = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, 1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, embeddings,positive):
        cat_embeddings = torch.cat((embeddings, positive),dim = -1)

        pre = F.relu(self.lin1(cat_embeddings))
        pre = F.dropout(pre,training=self.training)
        pre = self.lin2(pre)

        return pre

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads


        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        #self.output_layer = nn.Linear(num_heads * att_size, hidden_size)
        self.output_layer = nn.Linear(num_heads * att_size, 1)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x).view(orig_q_size[0],)

        return torch.softmax(x,dim=0)

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x