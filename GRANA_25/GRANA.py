#S is converted from numpy. The first array is index, the second array is the node to be calculated,
# and the third array is the weight
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv
from torch_scatter import scatter


class GRANA(MessagePassing):
    def __init__(self, in_dim_sn1, in_dim_sn2,  out_dim, attribute_dim, MTL_weight,
                 sn_model_type='GCN', attributed=True, dropout=0.1):
        super(GRANA, self).__init__(aggr='add')
        self.attributed = attributed
        self.model_type = sn_model_type

        if sn_model_type == 'GCN':
            self.sn1_model = GCNConv(in_dim_sn1, out_dim, bias=False)
            self.sn2_model = GCNConv(in_dim_sn2, out_dim, bias=False)
        else:
            self.sn1_model = GATConv(in_dim_sn1, out_dim, bias=False, dropout=dropout)
            self.sn2_model = GATConv(in_dim_sn2, out_dim, bias=False, dropout=dropout)

        # Parameter
        # parameters used in cross convolution
        self.cross_weight = torch.empty(out_dim * 2, out_dim)
        torch.nn.init.xavier_uniform_(self.cross_weight)
        self.cross_weight = nn.Parameter(self.cross_weight, requires_grad=True)

        # attribute reconstruction
        self.attr_linear = nn.Linear(out_dim, attribute_dim, bias=False)
        self.attr_bias1 = torch.empty(1, attribute_dim)
        torch.nn.init.xavier_uniform_(self.attr_bias1)
        self.attr_bias1 = nn.Parameter(self.attr_bias1, requires_grad=True)
        self.attr_bias2 = torch.empty(1, attribute_dim)
        torch.nn.init.xavier_uniform_(self.attr_bias2)
        self.attr_bias2 = nn.Parameter(self.attr_bias2, requires_grad=True)

        # structure reconstruction
        self.stru_bias1 = torch.empty(1, out_dim)
        torch.nn.init.xavier_uniform_(self.stru_bias1)
        self.stru_bias1 = nn.Parameter(self.stru_bias1, requires_grad=True)
        self.stru_bias2 = torch.empty(1, out_dim)
        torch.nn.init.xavier_uniform_(self.stru_bias2)
        self.stru_bias2 = nn.Parameter(self.stru_bias2, requires_grad=True)

        # multi-task learning weights
        self.MTL_weight = MTL_weight
        self.MTL_weight = nn.Parameter(self.MTL_weight, requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.sn1_model.reset_parameters()
        self.sn2_model.reset_parameters()
        self.attr_linear.reset_parameters()

    def forward(self, x1, edge_index1, x2, edge_index2, S1, S2):

        shape1 = S1.shape[1]
        shape2 = S2.shape[1]

        relu = nn.ReLU()

        # graph convolution
        if self.model_type == 'GCN':
            U1 = relu(self.sn1_model(x1, edge_index1))
            U2 = relu(self.sn2_model(x2, edge_index2))
        else:
            U1 = self.sn1_model(x1, edge_index1)
            U2 = self.sn2_model(x2, edge_index2)

        # cross layer
        cross_neighbor1 = U2[S1[1].type(torch.long)] * (S1[2].view([shape1, 1]))
        cross_neighbor1 = scatter(cross_neighbor1, S1[0].type(torch.long), dim=0, reduce='add').float()

        cross_neighbor2 = U1[S2[1].type(torch.long)] * (S2[2].view([shape2, 1]))
        cross_neighbor2 = scatter(cross_neighbor2, S2[0].type(torch.long), dim=0, reduce='add').float()

        x_out1 = torch.matmul(torch.cat(
            (U1, cross_neighbor1), 1), self.cross_weight)
        x_out2 = torch.matmul(torch.cat(
            (U2, cross_neighbor2), 1), self.cross_weight)

        x_out1_neighbor = x_out2[S1[1].type(torch.long)] * (S1[2].view([shape1, 1]))
        x_out1_neighbor = scatter(x_out1_neighbor, S1[0].type(torch.long), dim=0, reduce='add').float()
        x_out2_neighbor = x_out1[S2[1].type(torch.long)] * (S2[2].view([shape2, 1]))
        x_out2_neighbor = scatter(x_out2_neighbor, S2[0].type(torch.long), dim=0, reduce='add').float()

        x_out_neighbor = torch.cat((x_out1_neighbor, x_out2_neighbor), dim=0)

        # attribute reconstruction
        if self.attributed == True:
            x_out1_attr = self.attr_linear(x_out1) + self.attr_bias1
            x_out2_attr = self.attr_linear(x_out2) + self.attr_bias2

            x_out_attr = torch.cat((x_out1_attr, x_out2_attr), dim=0)
        else:
            x_out_attr = []

        # structure reconstruction
        x_out1_stru = x_out1 + self.stru_bias1
        x_out2_stru = x_out2 + self.stru_bias2
        x_out_stru = torch.cat((x_out1_stru, x_out2_stru), dim=0)

        return x_out1, x_out2, x_out_attr, x_out_stru, x_out_neighbor

    def stru_loss(self, emb1, emb2, sign):
        logsigmoid = nn.LogSigmoid()
        if isinstance(emb1, torch.Tensor):
            loss = -torch.mean(logsigmoid(torch.mul(
                sign, torch.sum(torch.mul(emb1, emb2), dim=1))))
        else:
            loss = torch.tensor(0.0).to(self.cross_weight.device)
        return loss

    def emb_loss(self, emb, emb_neighbor):
        if isinstance(emb, torch.Tensor):
            align_emb_loss = torch.mean(torch.norm(emb - emb_neighbor, p=2, dim=1))
        else:
            align_emb_loss = torch.tensor(0.0).to(self.cross_weight.device)

        return align_emb_loss

    def loss(self, loss_stack, state='in'):
        # loss with multi-task learning
        if state == 'in':
            total_loss = torch.sum(self.MTL_weight * loss_stack)
        else:
            total_loss = torch.sum(self.MTL_weight * loss_stack) + \
                   torch.sum(torch.log(1 / (2 * self.MTL_weight)))

        return total_loss

