import copy
import random
import math
from math import cos, pi
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
import pandas as pd
from torchmetrics.functional.classification import binary_accuracy, binary_auroc, binary_f1_score, \
    binary_average_precision, binary_roc
import os
from torch.nn import LayerNorm
from protein_pretrain import pretrain_init
from torchvision.ops import sigmoid_focal_loss
from base_model import forwardmodel, MultiHeadAttention, update_params, LSLRGradientDescentLearningRule, plot
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data,Batch
from collections import defaultdict
import pickle
directory=["./train_roc_curve_review_v9/","./zero_roc_curve_review_v9/"]


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Embedding):
        torch.nn.init.kaiming_uniform(m.weight, nonlinearity='relu')


def adjust_learning_rate(optimizer, current_step, max_step, warmup_step, lr_min=0.0, lr_max=0.1):
    if current_step < warmup_step:
        lr = lr_max * current_step / warmup_step
    else:
        lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * (current_step - warmup_step) / (max_step - warmup_step))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class GCN_DTIMAML(pl.LightningModule):
    def __init__(self, args):
        super(GCN_DTIMAML, self).__init__()
        self.K_shot = args.k_shot
        self.K_query = args.k_query
        self.batch_size = args.batch_size
        self.num_inner_steps = args.num_inner_steps

        self.weight_decay = args.weight_decay
        self.val_shot = args.val_shot
        self.MI_weight = 0.1
        self.con_weight = 0.1
        self.model = forwardmodel(args.protein_dim1, args.protein_dim2, args.protein_dim3, args.molecule_dim1,
                                  args.molecule_dim2, args.hidden_dim, args.hidden_dim2)
        self.meta_lr = args.meta_lr
        self.Attention=MultiHeadAttention(args.protein_dim3,attention_dropout_rate=args.attention_dropout_rate,num_heads=args.num_heads)
        self.task_lr = args.task_lr
        self.few_lr = args.few_lr
        self.total_epoch = args.total_epoch
        self.few_epoch = args.few_epoch
        self.iterations = args.iteration
        self.multi_step_loss_num_epochs = args.total_epoch
        self.enable_inner_loop_optimizable_bn_params = False
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=self.device,
                                                                    init_learning_rate=self.task_lr,
                                                                    total_num_inner_loop_steps=self.num_inner_steps)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.model.named_parameters()))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.meta_lr,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=4*self.iterations,eta_min=5e-6)
        return {"optimizer": optimizer,  "lr_scheduler": scheduler}


    def support_step(self,names_weights_copy,node_feat,label,edge_attr,edge_index,batch,protein_edge_index,protein_node_feat,step_idx):
        label = label.unsqueeze(1)
        label_loss = label.float()


        pred, protein_pos_penalty,protein_mi_loss,_= self.model(protein_node_feat,protein_edge_index, node_feat,edge_index, edge_attr, batch)
        sigmoid_pred = torch.sigmoid(pred)
        cls = F.binary_cross_entropy(sigmoid_pred, label_loss)
        loss = (1 - self.con_weight) * (cls) + (self.con_weight/2) * protein_pos_penalty + (self.con_weight/2)*protein_mi_loss

        grads = torch.autograd.grad(loss, names_weights_copy.values())
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     step_idx=step_idx)
        new_parm = []
        for param in names_weights_copy.values():
            new_parm.append(param.view(-1))
        new_parm = torch.cat(new_parm)
        vector_to_parameters(new_parm, self.model.parameters())


    def query_step(self,node_feat,label,edge_attr,edge_index,batch,protein_node_feat,protein_edge_index):

        label = label.unsqueeze(1)
        label_loss = label.float()

        pred, protein_pos_penalty,protein_mi_loss,protein_subgraph_emb = self.model(protein_node_feat,protein_edge_index, node_feat,
                                                                                      edge_index, edge_attr, batch)
        sigmoid_pred = torch.sigmoid(pred)
        cls=F.binary_cross_entropy(sigmoid_pred,label_loss)

        loss = (1 - self.con_weight) * (cls) + (self.con_weight/2) * protein_pos_penalty + (self.con_weight/2)*protein_mi_loss
        cls_show=F.binary_cross_entropy(sigmoid_pred,label_loss)
        return loss, cls_show, protein_pos_penalty,protein_mi_loss,sigmoid_pred, label, protein_subgraph_emb

    def training_step(self, batch, batch_idx):
        edge_index,edge_attr,node_feat,label,batch_batch,pdbID=self.new_seperate(batch)
        optim = self.optimizers()
        lr_scheduler=self.lr_schedulers()
        optim.zero_grad()
        preds = []
        labels = []
        weight_sumemb=[]
        total_loss = []
        total_cls = []
        total_pos_penalty = []
        total_miloss=[]
        old_parms = parameters_to_vector(self.model.parameters())
        for i in range(len(batch_batch)//2):
            task_loss = []
            task_cls = []
            task_mi_loss=[]
            task_pos_penalty = []
            protein_edge_index,protein_node_feat=pretrain_init(pdbID[i])
            protein_edge_index=protein_edge_index.to(self.device)
            protein_node_feat=protein_node_feat.to(self.device).detach()
            for step_idx in range(self.num_inner_steps):
                per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
                names_weights_copy = self.get_inner_loop_parameter_dict(self.model.named_parameters())
                self.support_step(names_weights_copy,node_feat[i*2],label[i*2],edge_attr[i*2],edge_index[i*2],batch_batch[i*2],protein_edge_index,protein_node_feat,step_idx)
                loss,cls,pos_penalty,mi_loss,pred,single_label,weight_emb=self.query_step(node_feat[i*2+1],label[i*2+1],edge_attr[i*2+1],edge_index[i*2+1],batch_batch[i*2+1],protein_node_feat,protein_edge_index)
                task_loss.append(per_step_loss_importance_vectors[step_idx] * loss)
                task_cls.append(cls.detach())
                task_pos_penalty.append(pos_penalty.detach())
                task_mi_loss.append(mi_loss.detach())
            task_loss = torch.sum(torch.stack(task_loss))
            task_cls = torch.sum(torch.stack(task_cls)) / len(task_cls)
            task_pos_penalty = torch.sum(torch.stack(task_pos_penalty)) / len(task_pos_penalty)
            task_mi_loss=torch.sum(torch.stack(task_mi_loss))/len(task_mi_loss)
            total_loss.append(task_loss)
            total_cls.append(task_cls)
            total_pos_penalty.append(task_pos_penalty)
            total_miloss.append(task_mi_loss)
            preds.append(pred.detach())
            weight_sumemb.append(weight_emb.detach())
            labels.append(single_label.detach())
            vector_to_parameters(old_parms, self.model.parameters())

        weight_sumemb=torch.cat(weight_sumemb,dim=0)
        loss_weight=self.Attention(weight_sumemb,weight_sumemb,weight_sumemb)
        '''loss_weight=[]
        for i in range(len(batch_batch) // 2):
            max_value, max_index = torch.max(batch_batch[i*2+1],dim=0)
            loss_weight.append(1/(max_value.item()+1))
        total_count = sum(loss_weight)
        loss_weight = [count / total_count for count in loss_weight]'''

        total_loss=torch.dot(torch.stack(total_loss,dim=0),torch.tensor(loss_weight).to(total_loss[0].device))
        #total_loss=torch.sum(loss_weight*torch.stack(total_loss,dim=0))
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        lr_scheduler.step()

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = binary_accuracy(preds, labels)
        auroc = binary_auroc(preds, labels)
        auprc = binary_average_precision(preds, labels)
        F1 = binary_f1_score(preds, labels)
        fpr, tpr, _ = binary_roc(preds, labels)
        plot(fpr.cpu(), tpr.cpu(), 'False Positive Rate', 'True Positive Rate', batch_idx, dir=directory[0])

        self.log("train_cls", sum(total_cls) / len(total_cls),sync_dist=True)
        self.log("train_pos_penalty", sum(total_pos_penalty) / len(total_pos_penalty),sync_dist=True)
        self.log("train_miloss", sum(total_miloss) / len(total_miloss), sync_dist=True)
        self.log("train_loss", total_loss,sync_dist=True)
        self.log("train_acc", acc,sync_dist=True)
        self.log("train_auroc", auroc,sync_dist=True)
        self.log("train_auprc", auprc,sync_dist=True)
        self.log("train_F1", F1,sync_dist=True)
        self.log("task_lr", torch.tensor(self.task_lr), sync_dist=True)
        return None

    def get_per_step_loss_importance_vector(self):
        loss_weights = np.ones(shape=(self.num_inner_steps)) * (
                1.0 / self.num_inner_steps)
        decay_rate = 1.0 / self.num_inner_steps / self.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.num_inner_steps
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.num_inner_steps - 1) * decay_rate),
            1.0 - ((self.num_inner_steps - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        return {
            name: param.to(device=self.device) for name, param in params
        }
    def new_seperate(self,batch):
        data_list = batch.to_data_list()
        edge_index, edge_attr, node_feat, label, batch_batch, pdbID=[],[],[],[],[],[]
        protein_index={}
        for index,data in enumerate(data_list):
            if data.protein_pdbID not in protein_index:
                protein_index[data.protein_pdbID]={}
                protein_index[data.protein_pdbID]["start"]=index
                protein_index[data.protein_pdbID]["end"] = index
            protein_index[data.protein_pdbID]["end"] = index
        for key,value in protein_index.items():
            start=value["start"]
            end=value["end"]
            shot_data=data_list[start:start+2*self.K_shot]
            query_data=data_list[start+2*self.K_shot:end+1]
            shot_batch=Batch.from_data_list(shot_data)
            query_batch=Batch.from_data_list(query_data)
            edge_index.extend([shot_batch.edge_index,query_batch.edge_index])
            edge_attr.extend([shot_batch.edge_attr,query_batch.edge_attr])
            node_feat.extend([shot_batch.x,query_batch.x])
            label.extend([shot_batch.y,query_batch.y])
            batch_batch.extend([shot_batch.batch,query_batch.batch])
            pdbID.append(key)
        return edge_index, edge_attr, node_feat, label, batch_batch, pdbID

    def validation_step(self, batch, batch_idx):
        node_feat, label, edge_attr, edge_index, val_batch = batch.x,batch.y,batch.edge_attr,batch.edge_index,batch.batch
        protein_edge_index,protein_node_feat=pretrain_init(batch.node_stores[0]["protein_pdbID"][0])
        protein_edge_index=protein_edge_index.to(self.device)
        protein_node_feat=protein_node_feat.to(self.device).detach()

        pred, protein_assignment = self.model(protein_node_feat, protein_edge_index, node_feat, edge_index, edge_attr,
                                              val_batch, mode="zero")
        pred = torch.sigmoid(pred).detach().cpu()
        label = label.unsqueeze(1).detach().cpu()


        return [pred,label] # ,few_loss,few_acc,few_auroc,few_auprc,few_F1]

    def validation_epoch_end(self, outputs):
        preds = []
        labels = []

        for single_output in outputs:
            preds.append(single_output[0])
            labels.append(single_output[1])

        preds = torch.cat(preds)
        labels = torch.cat(labels)
        label_loss = labels.float()
        loss = F.binary_cross_entropy(preds, label_loss)

        acc = binary_accuracy(preds, labels)
        auroc = binary_auroc(preds, labels)
        auprc = binary_average_precision(preds, labels)
        F1 = binary_f1_score(preds, labels)
        fpr, tpr, _ = binary_roc(preds, labels)
        plot(fpr.cpu(), tpr.cpu(), 'False Positive Rate', 'True Positive Rate', "", dir=directory[1])
        a = open(directory[1] + "result" + ".txt", mode="w")
        for pred_item, label_item in zip(preds.detach(), labels.detach()):
            a.writelines([str(pred_item.item()), "   ", str(label_item.item()), "\n"])
        a.close()

        self.log("zero_loss", loss)
        self.log("zero_acc", acc)
        self.log("zero_auroc", auroc)
        self.log("zero_auprc", auprc)
        self.log("zero_F1", F1)

    def test_step(self, batch,batch_idx):
        node_feat,label,edge_attr,edge_index,test_batch=batch.x,batch.y,batch.edge_attr,batch.edge_index,batch.batch
        protein_edge_index, protein_node_feat = pretrain_init(batch.node_stores[0]["protein_pdbID"][0])
        protein_node_feat=protein_node_feat.to(self.device).detach()
        protein_edge_index=protein_edge_index.to(self.device)

        pred, protein_assignment = self.model(protein_node_feat, protein_edge_index, node_feat, edge_index, edge_attr,
                                              test_batch, mode="zero")
        pred=torch.sigmoid(pred)
        label=label.unsqueeze(1)

        #auroc = binary_auroc(pred, label)
        number=max(batch.batch)+1

        #protein_pdb=batch.node_stores[0]["protein_pdbID"]
        #smiles=batch.node_stores[0]["smiles"]
        #zero_loss,zero_acc,zero_auroc,zero_auprc,zero_F1=self.zero_shot(node_feat,label,edge_attr,edge_index,molecule_batch,protein_node_feat, protein_edge_index,protein_batch,batch_idx)
        #few_loss,few_acc,few_auroc,few_auprc,few_F1=self.few_shot(protein_node_feat, protein_edge_index, node_feat, edge_index, edge_attr, edge_num,label,batch_,pdbID,batch_idx)

        return [pred,label,int(number.item())]
        #return [batch.node_stores[0]["protein_pdbID"][0],auroc.detach(),number]
    def test_epoch_end(self, outputs):
        preds=[]
        labels=[]
        protein_assignments=[]
        for single_output in outputs:
            preds.append(single_output[0])
            labels.append(single_output[1])
            protein_assignments.append(single_output[2])
        preds=torch.cat(preds)
        labels=torch.cat(labels)
        label_loss=labels.float()
        cal_pred=preds.clone()
        cal_label=labels.clone()
        with open('transductive_vector.pkl', 'wb') as f:
            pickle.dump(np.array(preds.detach().cpu()), f)

        '''if sum(label_loss)>0:
            loss=F.binary_cross_entropy(cal_pred,cal_label.float())

        acc = binary_accuracy(cal_pred,cal_label)
        auroc=binary_auroc(cal_pred,cal_label)
        auprc=binary_average_precision(cal_pred,cal_label)
        F1=binary_f1_score(cal_pred,cal_label)

        print("loss:", loss)
        print("acc:", acc)
        print("auroc:", auroc)
        print("auprc:", auprc)
        print("F1", F1)'''

        '''preds = defaultdict(list)
        labels = defaultdict(list)
        numbers = defaultdict(int)
        for single_output in outputs:
            preds[single_output[0]].append(single_output[1])
            labels[single_output[0]].append(single_output[2])
            numbers[single_output[0]]+=single_output[3]
        
        f=open("./result_per_protein","w")
        for protein_name in numbers.keys():
            cal_pred = torch.cat(preds[protein_name]).clone()
            cal_label = torch.cat(labels[protein_name]).clone()
            auroc = binary_auroc(cal_pred, cal_label)
            if float(auroc.item())==0.0:
                continue
            f.write("\t".join(map(str,[protein_name,auroc.item(),numbers[protein_name]]))+"\n")
        f.close()'''
    def zero_shot(self, protein_node_feat, protein_edge_index, node_feats, edge_indexs, edge_attrs, labeles, batchs,
                  batch_idx, pdbID):
        losses = []
        preds = []
        labels = []
        for (node_feat, edge_index, edge_attr, label, batch) in zip(node_feats, edge_indexs, edge_attrs, labeles,
                                                                    batchs):
            pred, protein_assignment= self.model(protein_node_feat, protein_edge_index, node_feat,
                                                                       edge_index, batch,mode="zero")
            pred = torch.sigmoid(pred)
            label = label.unsqueeze(1)
            label_loss = label.float()
            cls = F.binary_cross_entropy(pred, label_loss)

            losses.append(cls.detach())
            preds.append(pred.detach())
            labels.append(label.detach())
        pred = torch.cat(preds, dim=0)
        label = torch.cat(labels, dim=0)
        acc = binary_accuracy(pred, label)
        auroc = binary_auroc(pred, label)
        auprc = binary_average_precision(pred, label)
        F1 = binary_f1_score(pred, label)
        fpr, tpr, _ = binary_roc(pred, label)
        plot(fpr.cpu(), tpr.cpu(), 'False Positive Rate', 'True Positive Rate', pdbID, dir=directory[1])
        a = open(directory[1] + pdbID + "_" + str(batch_idx) + ".txt", mode="w")
        for pred_item, label_item in zip(pred.detach(), label.detach()):
            a.writelines([str(pred_item.item()), "   ", str(label_item.item()), "\n"])
        a.close()
        protein_assignment = np.array(protein_assignment.detach().cpu())
        np.savetxt(directory[1] + pdbID + "_" + str(batch_idx) + "protein_ass.txt", protein_assignment)
        return [sum(losses) / len(losses), acc, auroc, auprc, F1]

    def few_shot(self, protein_node_feat, protein_edge_index, node_feat, edge_index, edge_attr, edge_num, label, batch,
                 protein_name, batch_idx):
        local_optim = torch.optim.AdamW(self.model.parameters(), lr=self.few_lr, weight_decay=self.weight_decay)
        local_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=local_optim, T_max=self.few_epoch,
                                                                     eta_min=1e-6)
        local_optim.zero_grad()

        val_num = random.randint(0, len(edge_num) - 1)

        val_edge_index = edge_index[val_num]
        edge_index.pop(val_num)

        val_node_feat = node_feat[val_num]
        node_feat.pop(val_num)

        val_label = label[val_num].unsqueeze(1)
        label.pop(val_num)

        val_batch = batch[val_num]
        batch.pop(val_num)

        # losses,accs,aurocs,auprcs,F1s=[],[],[],[],[]
        best_auroc = 0
        best_auprc = 0
        best_F1 = 0
        best_val_loss = float("inf")
        best_acc = 0
        for i in range(self.few_epoch):
            loss_s = []
            for (train_node_feat, train_edge_index, train_label, train_batch) in zip(node_feat, edge_index, label,
                                                                                     batch):
                pred, protein_pos_penalty, molecule_pos_penalty, molecule_mi_loss, _ = \
                    self.model(protein_node_feat, protein_edge_index, train_node_feat, train_edge_index, train_batch)
                cls = F.binary_cross_entropy_with_logits(pred, train_label.unsqueeze(1).float())
                loss = (1 - self.con_weight) * (cls + self.MI_weight * molecule_mi_loss) + self.con_weight * (
                            protein_pos_penalty + molecule_pos_penalty)
                loss_s.append(loss)
            loss_s = sum(loss_s) / len(loss_s)
            local_optim.zero_grad()
            loss_s.backward()
            local_optim.step()
            local_scheduler.step()

            if i % 5 == 0 or i == self.few_epoch - 1:
                pred = self.model(protein_node_feat, protein_edge_index, val_node_feat, val_edge_index, val_batch,
                                  mode="val")
                val_loss = F.binary_cross_entropy_with_logits(pred, val_label.float())

                acc = binary_accuracy(pred, val_label)
                auroc = binary_auroc(pred, val_label)
                auprc = binary_average_precision(pred, val_label)
                F1 = binary_f1_score(pred, val_label)
                fpr, tpr, _ = binary_roc(pred, val_label)
                plot(fpr.cpu(), tpr.cpu(), 'False Positive Rate', 'True Positive Rate', batch_idx,
                     dir="./few_roc_curve/")
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                if auroc.detach().item() > best_auroc:
                    best_auroc = auroc.detach().item()
                if auprc.detach().item() > best_auprc:
                    best_auprc = auprc.detach().item()
                if F1.detach().item() > best_F1:
                    best_F1 = F1.detach().item()
                if acc.detach().item() > best_acc:
                    best_acc = acc.detach().item()
                '''aurocs.append(auroc.detach().item())
                auprcs.append(auprc.detach().item())
                F1s.append(F1.detach().item())
                losses.append(val_loss.detach().item())
                accs.append(acc.detach().item())
        plot(len(losses),losses,xlabel="epoch",ylabel="loss",protein_name=protein_name)
        plot(len(losses),aurocs,xlabel="epoch",ylabel="aurocs",protein_name=protein_name)
        plot(len(losses),auprcs,xlabel="epoch",ylabel="auprcs",protein_name=protein_name)
        plot(len(losses),F1s,xlabel="epoch",ylabel="F1s",protein_name=protein_name)'''
        return best_val_loss, best_acc, best_auroc, best_auprc, best_F1


if __name__ == "__main__":
    protein_graph("3eiy")
