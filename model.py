# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import collections
import time

import numpy as np
import torch
import math
import torch.nn as nn
import pytorch_lightning as pl
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
import torchmetrics
from copy import deepcopy
import torch.nn.functional as F
import csv
import os
import copy
import random
from pre_process import protein_datasets,read_protein,protein_interaction

import matplotlib.pyplot as plt
import wandb
from protein_pretrain import PolynomialDecayLR
import esm
from torch.nn.utils.convert_parameters import (vector_to_parameters,parameters_to_vector)

device=torch.device('cuda:0')


class early_stop:
    def __init__(self,delta,patience):
        self.delta=delta
        self.patience=patience
        self.count=0
        self.val_loss_last=float("inf")
        self.val_loss_last_last=float("inf")
    def judge(self,val_loss):
        if val_loss>self.val_loss_last_last+self.delta or val_loss>self.val_loss_last+self.delta:
            self.count+=1
            if self.count>=self.patience:
                return True
        elif val_loss<self.val_loss_last+self.delta and val_loss<self.val_loss_last_last+self.delta:
            self.count=0
        self.val_loss_last_last=self.val_loss_last
        self.val_loss_last=val_loss
        return False

class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MLP,self).__init__()    #
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(input_dim,hidden_dim)  # 第一个隐含层
        self.fc2 = torch.nn.Linear(hidden_dim,output_dim)  # 输出层

    def forward(self,din):
        # 前向传播， 输入值：din, 返回值 dout
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = F.softmax(self.fc2(dout), dim=1)  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return dout


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def split_batch(graphs, labels,k_query):
    support_graphs, query_graphs = graphs[:-1*k_query*2],graphs[-1*k_query*2:]
    support_labels, query_labels = labels[:-1*k_query*2],labels[-1*k_query*2:]

    return support_graphs, query_graphs, support_labels, query_labels

def split_batch_val_test(graphs, labels,val_num):
    support_graphs, query_graphs = graphs[:-1*val_num],graphs[-1*val_num:]
    support_labels, query_labels = labels[:-1*val_num],labels[-1*val_num:]

    return support_graphs, query_graphs, support_labels, query_labels



class DTIMAML(pl.LightningModule):
    def __init__(self,lr, num_inner_steps,n_layers,
                 num_heads,
                 hidden_dim,
                 dropout_rate,
                 intput_dropout_rate,
                 weight_decay,
                 ffn_dim,
                 dataset_name,
                 peak_lr,
                 end_lr,
                 edge_type,
                 multi_hop_max_dist,
                 attention_dropout_rate,
                 k_query,
                 val_val_batch_size,
                 val_train_batch_size,
                 warmup_updates,
                 tot_updates,
                 flag=False,
                 flag_m=3,
                 flag_step_size=1e-3,
                 flag_mag=1e-3,
                 ):
        """Inputs.

        proto_dim - Dimensionality of prototype feature space
        lr - Learning rate of the outer loop Adam optimizer
        lr_inner - Learning rate of the inner loop SGD optimizer
        lr_output - Learning rate for the output layer in the inner loop
        num_inner_steps - Number of inner loop updates to perform
        """
        super().__init__()
        self.molecule_model = Graphormer(n_layers,
                                num_heads,
                                hidden_dim,
                                dropout_rate,
                                intput_dropout_rate,
                                weight_decay,
                                ffn_dim,
                                dataset_name,
                                peak_lr,
                                end_lr,
                                edge_type,
                                multi_hop_max_dist,
                                attention_dropout_rate,
                                flag=flag,
                                flag_m=flag_m,
                                flag_step_size=flag_step_size,
                                flag_mag=flag_mag)
        self.molecule_MLP1=MLP(input_dim=ffn_dim,hidden_dim=512,output_dim=2)
        self.molecule_MLP2=MLP(input_dim=ffn_dim,hidden_dim=512,output_dim=1)
        '''self.protein_dataset=read_protein()
        self.protein_embedding=Protein_Embedding(input_dim_protein,transformer_emb_size_target,transformer_dropout_rate,transformer_n_layer_target,
                                            transformer_intermediate_size_target,transformer_num_attention_heads_target,transformer_attention_probs_dropout,
                                            transformer_hidden_dropout_rate)
        self.gcn=GCN(protein_name,args.transformer_emb_size_target,args.transformer_intermediate_size_target,args.transformer_emb_size_target)
        self.update_weight=nn.Linear(hidden_dim,1)'''
        '''if checkpoint_path!='':
            state_dict=torch.load(checkpoint_path)['state_dict']
            state_dict2 = collections.OrderedDict([(key[6:],value) for key,value in state_dict.items()])
            self.molecule_model.load_state_dict(state_dict2)
        self.protein_model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.protein_data,self.protein_interaction=read_protein()
        self.MLP=MLP(1280,hidden_dim)'''



        self.peak_lr=peak_lr
        self.end_lr=end_lr
        self.warmup_updates=warmup_updates
        self.tot_updates=tot_updates
        self.lr=lr
        self.num_inner_steps=num_inner_steps
        self.weight_decay=weight_decay
        self.k_query=k_query
        self.valtest_val_batch_size=val_val_batch_size
        self.valtest_train_batch_size=val_train_batch_size
        self.pretrain_file_dir="/amax/yxwang/GCN/meta-Graphormer/graphormer/model_temp/"
        self.pretrain_file_name=[]
        self.pretrain_file_num={}
        self.epoch=0

        for root,dirs,files in os.walk(self.pretrain_file_dir):
            for f in files:
                self.pretrain_file_name.append(f)
                self.pretrain_file_num[f[:-5]]=1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.molecule_model.parameters(), lr=self.lr)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
        lr_scheduler=PolynomialDecayLR(
            optimizer,
            warmup_updates=1000,
            tot_updates=5000,
            lr=1e-4,
            end_lr=1e-3,
            power=1.0)

        return [optimizer],[lr_scheduler]



    def run_model(self, graphs, labels):
        # Execute a model with given output layer weights and inputs

        molecule_emb = self.molecule_model(graphs)
        assign=self.molecule_MLP1(molecule_emb)
        Gsub=torch.mul(assign[0,:],molecule_emb)
        preds=self.molecule_MLP2(Gsub)

        labels=torch.stack(labels).to(device)
        label_loss=labels.float()
        loss=F.binary_cross_entropy_with_logits(preds,label_loss)
        acc = torchmetrics.functional.accuracy(preds,labels,average='micro')
        auroc=torchmetrics.functional.auroc(preds,labels,average='micro')
        return loss,acc,auroc


    def adapt_few_shot(self, support_graphs,support_labels,query_graphs, query_labels,protein_name,batch_idx,mode):

        # Create inner-loop model and optimizer

        '''for pretrain_file_name in self.pretrain_file_name:
            if protein_name in pretrain_file_name:
                if self.pretrain_file_num[pretrain_file_name]>0:
                    checkpoint=torch.load(pretrain_file_name,map_location=device)['state_dict']
                    local_model.load_state_dict(checkpoint)
                    self.pretrain_file_num[pretrain_file_name]-=1
                break'''
        '''if self.pretrain_file_num[protein_name]==1:
            checkpoint=torch.load("./model_temp/"+protein_name+".ckpt",map_location=device)['state_dict']
            local_model.load_state_dict(checkpoint)
            self.pretrain_file_num[protein_name]-=1
        else:
            checkpoint=torch.load("./model_temp/"+protein_name+".pth",map_location=device)['state_dict']
            local_model.load_state_dict(checkpoint)'''
        molecule_local_optim = torch.optim.AdamW(self.molecule_model.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        molecule_MLP1_local_optim=torch.optim.AdamW(self.molecule_MLP1.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        molecule_MLP2_local_optim=torch.optim.AdamW(self.molecule_MLP2.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        molecule_scheduler=PolynomialDecayLR(
            self.molecule_local_optim,
            warmup_updates=100,
            tot_updates=300,
            lr=self.peak_lr,
            end_lr=self.end_lr,
            power=1.0)


        val_losses,val_aurocs=[],[]
        best_auroc=torch.tensor(0).to(device)
        # Optimize inner loop model on support set
        for m in range(self.num_inner_steps):
            # Determine loss on the support set
            loss,_,_ = self.run_model(support_graphs, support_labels)
            # Calculate gradients and perform inner loop update
            loss.backward()

            molecule_local_optim.step()
            molecule_MLP1_local_optim.step()
            molecule_MLP2_local_optim.step()
            molecule_scheduler.step()
            # Update output layer via SGD
            # Reset gradients
            molecule_local_optim.zero_grad()
            molecule_MLP1_local_optim.zero_grad()
            molecule_MLP2_local_optim.zero_grad()

            val_loss,val_acc,val_auroc,=self.run_model(query_graphs, query_labels)
            val_losses.append(val_loss.detach().item())
            val_aurocs.append(val_auroc.detach().item())

            if m>=self.num_inner_steps/2 :
                if val_auroc>best_auroc:
                    best_auroc=val_auroc
                    best_parm=(parameters_to_vector(self.molecule_model.parameters()),parameters_to_vector(self.molecule_MLP1.parameters()),parameters_to_vector(self.molecule_MLP2.parameters()))

        x=np.arange(self.num_inner_steps).tolist()
        save_fig_path="./inner_loss_fig/val_loss_"+str(batch_idx)+"_"+str(protein_name)+".png"
        plt.plot(x,val_losses,',-',color = 'g',label="loss")#o-:圆形
        plt.xlabel("inner_step")#横坐标名字
        plt.ylabel("val_loss")
        plt.savefig(save_fig_path,dpi=300)
        plt.close()
        save_fig_path1="./inner_loss_fig/auroc_"+str(batch_idx)+"_"+str(protein_name)+".png"
        plt.plot(x,val_aurocs,',-',color = 'g',label="loss")#o-:圆形
        plt.xlabel("inner_step")#横坐标名字
        plt.ylabel("val_auroc")
        plt.savefig(save_fig_path1,dpi=300)
        plt.close()
        #checkpoint = {'state_dict': local_model.state_dict(),'optimizer' :local_optim.state_dict()}
        #torch.save(checkpoint,"./model_temp/"+protein_name+".pth")
        return best_parm


    def training_step(self, batch,batch_idx):

        #self.get_protein(list(set(batch.protein_names)))

        aurocs = []
        accs=[]
        losses = []
        losses_q = torch.tensor([0.0]).to(device)
        molecule_old_params = parameters_to_vector(self.molecule_model.parameters())
        molecule_MLP1_old_params=parameters_to_vector(self.molecule_MLP1.parameters())
        molecule_MLP2_old_params=parameters_to_vector(self.molecule_MLP2.parameters())

        self.molecule_model.zero_grad()
        self.molecule_MLP1.zero_grad()
        self.molecule_MLP2.zero_grad()
        self.molecule_MLP3.zero_grad()


        # Determine gradients for batch of tasks
        for i,task_batch in enumerate(batch):
            graphs, labels,protein_name= task_batch
            support_graphs, query_graphs, support_labels, query_labels= split_batch(graphs, labels,self.k_query)
            # Perform inner loop adaptation

            support_parm=self.adapt_few_shot(support_graphs,support_labels,query_graphs, query_labels,protein_name,batch_idx,mode="train")
            # Determine loss of query set
            support_mol_parm,support_molMLP1_parm,support_molMLP2_parm=support_parm
            vector_to_parameters(support_mol_parm, self.molecule_model.parameters())
            vector_to_parameters(support_molMLP1_parm,self.molecule_MLP1.parameters())
            vector_to_parameters(support_molMLP2_parm,self.molecule_MLP2.parameters())

            loss,acc,auroc = self.run_model(query_graphs, query_labels)
            # Calculate gradients for query set loss
            if i == 0:
                losses_q = loss
            else:
                losses_q = torch.cat((losses_q, loss), 0)

            aurocs.append(auroc.detach())
            losses.append(loss.detach())
            accs.append(acc.detach())
            vector_to_parameters(molecule_old_params, self.molecule_model.parameters())
            vector_to_parameters(molecule_MLP1_old_params,self.molecule_MLP1.parameters())
            vector_to_parameters(molecule_MLP2_old_params,self.molecule_MLP2.parameters())

        losses_q = torch.sum(losses_q)
        loss_q = losses_q / i

        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()
        self.lr_schedulers().step()
        # Perform update of base model

        self.log("train_loss", sum(losses) / len(losses))
        self.log("train_acc" , sum(accs) / len(accs))
        self.log("train_auroc",sum(aurocs)/len(aurocs))

        return None  # Returning None means we skip the default training optimizer steps by PyTorch Lightning


    def plot(self,epoch,y,xlabel,ylabel,protein_name):
        x=np.arange(epoch).tolist()
        plt.plot(x,y,',-',color = 'g')#o-:圆形
        plt.xlabel(xlabel)#横坐标名字
        plt.ylabel(ylabel)
        plt.savefig("./val_loss_fig/"+str(self.epoch)+ylabel+protein_name+".png",dpi=300)
        plt.close()

    def validation_step(self, batch,batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.molecule_model.zero_grad()
        self.molecule_MLP1.zero_grad()
        self.molecule_MLP2.zero_grad()

        random.seed(42)
        graphs,labels,protein_names= batch.graphs,batch.labels,batch.protein_names
        val_num=int(len(graphs)*0.2)
        support_graphs, query_graphs, support_labels, query_labels= split_batch_val_test(graphs, labels,val_num)


        index_list=np.arange(0,len(support_graphs))
        random.shuffle(index_list)
        new_support_graphs=[]
        new_support_labels=[]
        for index in index_list:
            new_support_graphs.append(support_graphs[index])
            new_support_labels.append(support_labels[index])

        index_list1=np.arange(0,len(query_labels))
        random.shuffle(index_list1)
        new_query_graphs=[]
        new_query_labels=[]
        for index1 in index_list1:
            new_query_graphs.append(query_graphs[index1])
            new_query_labels.append(query_labels[index1])

        index_list2=np.arange(0,len(labels))
        random.shuffle(index_list2)
        new_graphs=[]
        new_labels=[]
        for index2 in index_list2:
            new_graphs.append(graphs[index2])
            new_labels.append(labels[index2])

        molecule_old_params = parameters_to_vector(self.molecule_model.parameters())
        molecule_MLP1_old_params=parameters_to_vector(self.molecule_MLP1.parameters())
        molecule_MLP2_old_params=parameters_to_vector(self.molecule_MLP2.parameters())

        molecule_local_optim = torch.optim.AdamW(self.molecule_model.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        molecule_MLP1_local_optim=torch.optim.AdamW(self.molecule_MLP1.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        molecule_MLP2_local_optim=torch.optim.AdamW(self.molecule_MLP2.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        molecule_scheduler=PolynomialDecayLR(
            self.molecule_local_optim,
            warmup_updates=100,
            tot_updates=300,
            lr=self.peak_lr,
            end_lr=self.end_lr,
            power=1.0)

        val_losses,val_accs,val_aurocs,=[],[],[]
        losses_q = torch.tensor([0.0]).to(device)
        epoch=200
        best_auroc=torch.tensor(0).to(device)
        best_acc=torch.tensor(0).to(device)
        best_loss=torch.tensor(float("inf")).to(device)
        for m in range(epoch):
            for i in range(len(new_support_graphs)//self.valtest_train_batch_size):
                if i==(len(new_support_graphs)//self.valtest_train_batch_size)-1:
                    support_graphs_per_batch=new_support_graphs[i*self.valtest_train_batch_size:]
                    support_labels_per_batch=new_support_labels[i*self.valtest_train_batch_size:]
                else:
                    support_graphs_per_batch=new_support_graphs[i*self.valtest_train_batch_size:(i+1)*self.valtest_train_batch_size]
                    support_labels_per_batch=new_support_labels[i*self.valtest_train_batch_size:(i+1)*self.valtest_train_batch_size]

                loss,_,_ = self.run_model(support_graphs_per_batch, support_labels_per_batch)
                if i == 0:
                    losses_q = loss
                else:
                    losses_q = torch.cat((losses_q, loss), 0)
            loss=torch.sum(losses_q)/i
            loss.backward()
            molecule_local_optim.step()
            molecule_MLP1_local_optim.step()
            molecule_MLP2_local_optim.step()
            molecule_scheduler.step()


            molecule_local_optim.zero_grad()
            molecule_MLP1_local_optim.zero_grad()
            molecule_MLP2_local_optim.zero_grad()

            for j in range(len(new_query_graphs)//self.valtest_val_batch_size):
                if j==(len(new_query_graphs)//self.valtest_val_batch_size)-1:
                    query_graphs_per_batch=new_query_graphs[i*self.valtest_val_batch_size:]
                    query_labels_per_batch=new_query_labels[i*self.valtest_val_batch_size:]
                else:
                    query_graphs_per_batch=new_query_graphs[i*self.valtest_val_batch_size:(i+1)*self.valtest_val_batch_size]
                    query_labels_per_batch=new_query_labels[i*self.valtest_val_batch_size:(i+1)*self.valtest_val_batch_size]
                val_loss,val_acc,val_auroc,=self.run_model(query_graphs_per_batch, query_labels_per_batch)
                val_losses.append(val_loss.detach().item())
                val_accs.append(val_acc.detach().item())
                val_aurocs.append(val_auroc.detach().item())

            if m>=epoch/2 :
                if val_auroc>best_auroc:
                    best_auroc=val_auroc
                    best_acc=val_acc
                    best_loss=val_loss
                    best_parm=(parameters_to_vector(self.molecule_model.parameters()),parameters_to_vector(self.molecule_MLP1.parameters()),parameters_to_vector(self.molecule_MLP2.parameters()))

        self.plot(epoch,val_losses,"val_epoch","val_loss",protein_names[0])
        self.plot(epoch,val_accs,"val_epoch","val_acc",protein_names[0])
        self.plot(epoch,val_aurocs,"val_epoch","val_auroc",protein_names[0])


        self.log(protein_names[0]+"_acc",best_acc)
        self.log(protein_names[0]+"_auroc",best_auroc)

        vector_to_parameters(molecule_old_params, self.molecule_model.parameters())
        vector_to_parameters(molecule_MLP1_old_params,self.molecule_MLP1.parameters())
        vector_to_parameters(molecule_MLP2_old_params,self.molecule_MLP2.parameters())

        inductive_losses,inductive_accs,inductive_aurocs=[],[],[]
        for j in range(len(new_graphs)//self.valtest_val_batch_size):
            if j==(len(new_graphs)//self.valtest_val_batch_size)-1:
                graphs_per_batch=new_graphs[i*self.valtest_val_batch_size:]
                labels_per_batch=new_labels[i*self.valtest_val_batch_size:]
            else:
                graphs_per_batch=new_graphs[i*self.valtest_val_batch_size:(i+1)*self.valtest_val_batch_size]
                labels_per_batch=new_labels[i*self.valtest_val_batch_size:(i+1)*self.valtest_val_batch_size]
            val_loss,val_acc,val_auroc,=self.run_model(graphs_per_batch, labels_per_batch)
            inductive_losses.append(val_loss.detach().item())
            inductive_accs.append(val_acc.detach().item())
            inductive_aurocs.append(val_auroc.detach().item())

        inductive_loss=sum(inductive_losses) / len(inductive_losses)
        inductive_acc=sum(inductive_accs)/len(inductive_accs)
        inductive_auroc=sum(inductive_aurocs)/len(inductive_aurocs)

        self.log(protein_names[0]+"_inductive_loss",inductive_loss)
        self.log(protein_names[0]+"_inductive_acc",inductive_acc)
        self.log(protein_names[0]+"_inductive_auroc",inductive_auroc)

        torch.set_grad_enabled(False)
        return [best_loss,best_acc,best_auroc]

    def validation_epoch_end(self, outputs):
        loss=0
        acc=0
        auroc=0
        for single_output in outputs:
            loss+=single_output[0]
            acc+=single_output[1]
            auroc+=single_output[2]
        self.log("val_loss", loss/len(outputs))
        self.log("val_auroc",acc/len(outputs))
        self.log("val_auprc",auroc/len(outputs))

    def test_step(self,batch,batch_idx):

        torch.set_grad_enabled(True)

        random.seed(42)
        graphs,labels,protein_names= batch.graphs,batch.labels,batch.protein_names
        test_num=int(len(graphs)*0.2)
        support_graphs, query_graphs, support_labels, query_labels= split_batch_val_test(graphs, labels,test_num)

        index_list=np.arange(0,len(support_graphs))
        random.shuffle(index_list)
        new_support_graphs=[]
        new_support_labels=[]
        for index in index_list:
            new_support_graphs.append(support_graphs[index])
            new_support_labels.append(support_labels[index])

        index_list1=np.arange(0,len(query_labels))
        random.shuffle(index_list1)
        new_query_graphs=[]
        new_query_labels=[]
        for index1 in index_list1:
            new_query_graphs.append(query_graphs[index1])
            new_query_labels.append(query_labels[index1])

        index_list2=np.arange(0,len(labels))
        random.shuffle(index_list2)
        new_graphs=[]
        new_labels=[]
        for index2 in index_list2:
            new_graphs.append(graphs[index2])
            new_labels.append(labels[index2])

        local_model = deepcopy(self.molecule_model)
        local_model.train()
        local_optim = torch.optim.AdamW(local_model.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        inner_scheduler=PolynomialDecayLR(
            local_optim,
            warmup_updates=100,
            tot_updates=300,
            lr=self.peak_lr,
            end_lr=self.end_lr,
            power=1.0)
        local_optim.zero_grad()
        test_train_losses,test_test_losses,test_test_aurocs,test_test_auprcs=[],[],[],[]
        #earlystop=early_stop(delta=0.05,patience=2)
        epoch=0
        max_auroc=0
        for m in range(800):
            inloss=[]
            for i in range(len(new_support_graphs)//self.valtest_train_batch_size+1):
                if i==(len(new_support_graphs)//self.valtest_train_batch_size):
                    support_graphs_per_batch=new_support_graphs[i*self.valtest_train_batch_size:]
                    support_labels_per_batch=new_support_labels[i*self.valtest_train_batch_size:]
                else:
                    support_graphs_per_batch=new_support_graphs[i*self.valtest_train_batch_size:(i+1)*self.valtest_train_batch_size]
                    support_labels_per_batch=new_support_labels[i*self.valtest_train_batch_size:(i+1)*self.valtest_train_batch_size]

                loss,acc = self.run_model(local_model, support_graphs_per_batch, support_labels_per_batch)

                loss.backward()
                local_optim.step()
                local_optim.zero_grad()
                inloss.append(loss.detach().item())
                del loss,acc
            test_train_losses.append(sum(inloss)/len(inloss))
            test_test_loss,test_test_auroc,test_test_auprc=self.val_test_(local_model,new_query_graphs,new_query_labels,self.valtest_val_batch_size)
            test_test_losses.append(test_test_loss.item())
            test_test_aurocs.append(test_test_auroc.item())
            test_test_auprcs.append(test_test_auprc.item())
            if test_test_auroc.item()>max_auroc:
                max_auroc=test_test_auroc.item()
            epoch+=1
            #if m>20 and earlystop.judge(test_test_loss):
            #    break
            inner_scheduler.step()
        self.plot(epoch,test_train_losses,"test_epoch","test_train_loss",protein_names[0])
        self.plot(epoch,test_test_losses,"test_epoch","test_test_loss",protein_names[0])
        self.plot(epoch,test_test_aurocs,"test_epoch","test_test_auroc",protein_names[0])
        self.plot(epoch,test_test_auprcs,"test_epoch","test_test_auprc",protein_names[0])

        loss,auroc,auprc=self.val_test_(local_model,new_query_graphs,new_query_labels,self.valtest_val_batch_size)
        self.log(protein_names[0]+"_loss",loss)
        self.log(protein_names[0]+"_auroc",auroc)
        self.log(protein_names[0]+"_auprc",auprc)

        inductive_loss,inductive_auroc,inductive_auprc=self.val_test_(self.molecule_model,new_graphs,new_labels,self.valtest_val_batch_size)
        self.log(protein_names[0]+"_inductive_loss",inductive_loss)
        self.log(protein_names[0]+"_inductive_auroc",inductive_auroc)
        self.log(protein_names[0]+"_inductive_auprc",inductive_auprc)

        torch.set_grad_enabled(False)
        print("max_auroc",max_auroc)



class Graphormer(nn.Module):

    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        weight_decay,
        ffn_dim,
        dataset_name,
        peak_lr,
        end_lr,
        edge_type,
        multi_hop_max_dist,
        attention_dropout_rate,
        flag=False,
        flag_m=3,
        flag_step_size=1e-3,
        flag_mag=1e-3,
    ):
        super().__init__()
        #self.writer = SummaryWriter('runs/'+dataset_name+'/lr'+str(peak_lr))


        self.num_heads = num_heads
        if dataset_name == 'ZINC':
            self.atom_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(64, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(
                    40 * num_heads * num_heads, 1)
            self.spatial_pos_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
        else:
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
        self.dataset_name = dataset_name

        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.automatic_optimization = not self.flag
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        self.downstream_out_proj = nn.Linear(hidden_dim,1)

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

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

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

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

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
