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


class t_sne:
    def __init__(self):
        # 设置散点形状
        self.maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
        # 设置散点颜色
        self.colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
                  'hotpink']
        # 图例名称
        self.Label_Com = ['a', 'b', 'c', 'd']
        # 设置字体格式
        self.font1 = {'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 32,
                 }
        self.feat = torch.rand(128, 1024)  # 128个特征，每个特征的维度为1024
        label_test1 = [0 for index in range(40)]
        label_test2 = [1 for index in range(40)]
        label_test3 = [2 for index in range(48)]

        label_test = np.array(label_test1 + label_test2 + label_test3)
        print(label_test)
        print(label_test.shape)

        fig = plt.figure(figsize=(10, 10))

        plotlabels(visual(feat), label_test, '(a)')

        plt.show(fig)
    def visual(self,feat):
    # t-SNE的最终结果的降维与可视化
        ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
        x_ts = ts.fit_transform(feat)
        print(x_ts.shape)  # [num, 2]
        x_min, x_max = x_ts.min(0), x_ts.max(0)
        x_final = (x_ts - x_min) / (x_max - x_min)
        return x_final

    def plotlabels(self,S_lowDWeights, Trure_labels, name):
        True_labels = Trure_labels.reshape((-1, 1))
        S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
        S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
        print(S_data)
        print(S_data.shape)  # [num, 3]

        for index in range(3):  # 假设总共有三个类别，类别的表示为0,1,2
            X = S_data.loc[S_data['label'] == index]['x']
            Y = S_data.loc[S_data['label'] == index]['y']
            plt.scatter(X, Y, cmap='brg', s=100, marker=self.maker[index], c=self.colors[index], edgecolors=self.colors[index], alpha=0.65)

            plt.xticks([])  # 去掉横坐标值
            plt.yticks([])  # 去掉纵坐标值

        plt.title(name, fontsize=32, fontweight='normal', pad=20)


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
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, total_num_inner_loop_steps,init_learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
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
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
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
        #self.molecule_subgraph=GIBGCN(hidden_dim2)

        self.cat_MLP=MLP(2*hidden_dim,hidden_dim2,1)
        self.fc1 = Linear(protein_dim3, protein_dim3)
        self.fc2 = Linear(protein_dim3, hidden_dim)
        #self.molecule_discriminator = Discriminator(hidden_dim2)
        #self.protein_discriminator=Discriminator(protein_dim3)
        #self.discriminator_inner_loop=20
        #self.discriminator_lr=0.01

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
            #protein_mi_loss=self.sum_loss(self.protein_discriminator,protein_graph_embedding,protein_subgraph_emb)
            return pred,protein_pos_penalty,torch.mean(torch.cat([protein_subgraph_emb,molecule_embedding],dim=1),dim=0,keepdim=True)
        else:
            return pred,protein_assignment

    def sum_loss(self,discriminator,all_graph_embedding,subgraph_embedding):
        #optimizer_local = torch.optim.Adam(discriminator.parameters(), lr=0.01)
        #local_losses=[]
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

        #Donsker
        mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(margin)))
        #JSD
        #mi_est = -torch.mean(F.softplus(-joint)) - torch.mean(F.softplus(-margin)+margin)
        #mi_est = torch.mean(joint**2) - 0.5* torch.mean((torch.sqrt(margin**2)+1.0)**2)
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
        self.norm1=BatchNorm1d(hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.norms=torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,add_self_loops=True))
            self.norms.append(BatchNorm1d(hidden_channels))
        self.conv2 = GCNConv(hidden_channels, out_channels,add_self_loops=True)
        self.norm2=BatchNorm1d(out_channels)
    def forward(self, x, edge_index):
        x=self.conv1(x,edge_index)
        x=self.norm1(x)
        x = F.dropout(F.relu(x),0.5,training=self.training)
        for conv,norm in zip(self.convs,self.norms):
            x=conv(x,edge_index)
            x=norm(x)
            x = F.dropout(F.relu(x),0.5,training=self.training)
        x=self.conv2(x,edge_index)
        x=self.norm2(x)
        return x

class moleculeGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_layers=3):
        super(moleculeGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.norm1 = BatchNorm1d(hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=True))
            self.norms.append(BatchNorm1d(hidden_channels))
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)
        self.norm2 = BatchNorm1d(out_channels)

        self.pool = global_mean_pool


    def forward(self, x,edge_index,edge_attr,batch):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.dropout(F.relu(x), 0.5, training=self.training)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.dropout(F.relu(x), 0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        batch = torch.sub(input=batch, other=batch[0])
        x = self.pool(x, batch)

        return x
class GIBGCN(torch.nn.Module):
    def __init__(self, protein_dim3,hidden_dim):
        super(GIBGCN, self).__init__()

        #attention = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1)
        self.cluster1 = Linear(protein_dim3+hidden_dim, protein_dim3)
        self.cluster=nn.ModuleList()
        for _ in range(1):
            self.cluster.append(Linear(protein_dim3, protein_dim3))
        self.cluster2 = Linear(protein_dim3, 2)
        self.mse_loss = nn.MSELoss()
        self.fc1=Linear(protein_dim3,protein_dim3)
        self.fc2=Linear(protein_dim3,hidden_dim)

    def assignment(self,x):
        x = F.relu(self.cluster1(x))
        for cluster in self.cluster:
            x=cluster(x)
            x=F.relu(x)
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
        #self.norm2=LayerNorm(hidden_size)
        #init_params(self.lin1, n_layers=2)
        #init_params(self.lin2, n_layers=2)
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, embeddings,positive):
        cat_embeddings = torch.cat((embeddings, positive),dim = -1)

        pre = F.relu(self.lin1(cat_embeddings))
        pre = F.dropout(pre,training=self.training)
        pre = self.lin2(pre)

        return pre

class Graphormer(nn.Module):

    def __init__(
            self,
            n_layers,
            num_heads,
            hidden_dim,
            output_dim,
            dropout_rate,
            intput_dropout_rate,
            weight_decay,
            ffn_dim,
            edge_type,
            multi_hop_max_dist,
            attention_dropout_rate,
    ):
        super().__init__()
        #self.writer = SummaryWriter('runs/'+dataset_name+'/lr'+str(peak_lr))


        self.num_heads = num_heads
        self.atom_encoder = nn.Embedding(
                512 * 9 + 1, hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(
                512 * 3 + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(
                    128 * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(
                512, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
                512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.loss_fn = F.binary_cross_entropy_with_logits

        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist


        self.hidden_dim = hidden_dim
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        self.downstream_out_proj = nn.Linear(hidden_dim,output_dim)

    def forward(self, batched_data, perturb=None):


        attn_bias=torch.stack([batched_data[i]['attn_bias'] for i in range(len(batched_data))],0).to(device)
        spatial_pos=torch.stack([batched_data[i]['spatial_pos'] for i in range(len(batched_data))],0).to(device)
        x=torch.stack([batched_data[i]['x'] for i in range(len(batched_data))],0).to(device)
        in_degree=torch.stack([batched_data[i]['in_degree'] for i in range(len(batched_data))],0).to(device)
        out_degree=torch.stack([batched_data[i]['out_degree'] for i in range(len(batched_data))],0).to(device)
        edge_input=torch.stack([batched_data[i]['edge_input'] for i in range(len(batched_data))],0).to(device)
        attn_edge_type=torch.stack([batched_data[i]['attn_edge_type'] for i in range(len(batched_data))],0).to(device)



        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(
                3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) /
                          (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(
                attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,:, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token

        node_feature = self.atom_encoder(x).sum(dim=-2)           # [n_graph, n_node, n_hidden]
        if self.flag and perturb is not None:
            node_feature += perturb

        node_feature = node_feature + \
                       self.in_degree_encoder(in_degree) + \
                       self.out_degree_encoder(out_degree)

        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        output = self.downstream_out_proj(output[:, 0, :])

        return output

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
