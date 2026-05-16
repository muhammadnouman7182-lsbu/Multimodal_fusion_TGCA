# Base / Native
import csv
from collections import Counter
import copy
import json
import functools
import gc
import logging
import math
import os
import pdb
import pickle
import random
import sys
import tables
import time
from tqdm import tqdm

# Numerical / Array
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.utils.data import DataLoader

try:
    import timm
except ImportError:
    pass
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GatedGraphConv, GATConv
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.transforms.normalize_features import NormalizeFeatures

# Env
from fusion import *
from options import parse_args
from utils import *


################
# Network Utils
################
def define_net(opt, k):
    net = None
    act = define_act_layer(act_type=opt.act_type)
    init_max = True if opt.init_type == "max" else False

    mode = opt.mode
    if mode == "A": mode = "path"
    if mode == "B": mode = "graph"
    if mode == "C": mode = "omic"
    if mode == "AB": mode = "pathgraph"
    if mode == "AC": mode = "pathomic"
    if mode == "BC": mode = "graphomic"
    if mode == "ABC": mode = "pathgraphomic"

    if mode == "path":
        if opt.use_uni:
            net = UNINet(path_dim=opt.path_dim, act=act, num_classes=opt.label_dim)
        else:
            net = get_vgg(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim)
    elif mode == "graph":
        net = GraphNet(grph_dim=opt.grph_dim, dropout_rate=opt.dropout_rate, GNN=opt.GNN, use_edges=opt.use_edges, pooling_ratio=opt.pooling_ratio, act=act, label_dim=opt.label_dim, init_max=init_max)
    elif mode == "omic":
        if opt.use_transformer:
            net = GenosTransformer(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim)
        else:
            net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=init_max)
    elif mode == "graphomic":
        net = GraphomicNet(opt=opt, act=act, k=k)
    elif mode == "pathomic":
        net = PathomicNet(opt=opt, act=act, k=k)
    elif mode == "pathgraph":
        net = PathgraphNet(opt=opt, act=act, k=k)
    elif mode == "pathgraphomic":
        net = PathgraphomicNet(opt=opt, act=act, k=k)
    elif mode == "pathpath":
        net = PathpathNet(opt=opt, act=act, k=k)
    elif mode == "graphgraph":
        net = GraphgraphNet(opt=opt, act=act, k=k)
    elif mode == "omicomic":
        net = OmicomicNet(opt=opt, act=act, k=k)
    else:
        raise NotImplementedError('mode [%s] is not implemented' % opt.mode)
    return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)


def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def define_reg(opt, model):
    loss_reg = None
    
    if opt.reg_type == 'none':
        loss_reg = 0
    elif opt.reg_type == 'path':
        loss_reg = regularize_path_weights(model=model)
    elif opt.reg_type == 'mm':
        loss_reg = regularize_MM_weights(model=model)
    elif opt.reg_type == 'all':
        loss_reg = regularize_weights(model=model)
    elif opt.reg_type == 'omic':
        loss_reg = regularize_MM_omic(model=model)
    else:
        raise NotImplementedError('reg method [%s] is not implemented' % opt.reg_type)
    return loss_reg


def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
       scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
       scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
       scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer


def define_bifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=64, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion':
        fusion = BilinearFusion(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, dim1=dim1, dim2=dim2, scale_dim1=scale_dim1, scale_dim2=scale_dim2, mmhid=mmhid, dropout_rate=dropout_rate)
    elif fusion_type == 'crossattn':
        fusion = CrossAttnFusion(path_dim=dim1, omic_dim=dim2, hidden=mmhid)
    elif fusion_type == 'concat':
        fusion = ConcatFusion(dim1=dim1, dim2=dim2, mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion


def define_trifusion(fusion_type, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=3, dim1=32, dim2=32, dim3=32, scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=96, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion_A' or fusion_type == 'pofusion':
        fusion = TrilinearFusion_A(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, gate3=gate3, dim1=dim1, dim2=dim2, dim3=dim3, scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3, mmhid=mmhid, dropout_rate=dropout_rate)
    elif fusion_type == 'pofusion_B':
        fusion = TrilinearFusion_B(skip=skip, use_bilinear=use_bilinear, gate1=gate1, gate2=gate2, gate3=gate3, dim1=dim1, dim2=dim2, dim3=dim3, scale_dim1=scale_dim1, scale_dim2=scale_dim2, scale_dim3=scale_dim3, mmhid=mmhid, dropout_rate=dropout_rate)
    elif fusion_type == 'crossattn':
        class TriCrossAttnWrapper(nn.Module):
            def __init__(self, path_dim, omic_dim, hidden):
                super().__init__()
                self.core = CrossAttnFusion(path_dim=path_dim, omic_dim=omic_dim, hidden=hidden)
            def forward(self, v1, v2, v3):
                # v1=path, v2=grph, v3=omic -> map to CrossAttn(path_emb, omic_emb, grph_emb)
                return self.core(v1, v3, v2)
        fusion = TriCrossAttnWrapper(path_dim=dim1, omic_dim=dim3, hidden=mmhid)
    elif fusion_type == 'concat':
        # Simple concat for 3 modalities
        class TripleConcat(nn.Module):
            def __init__(self, d1, d2, d3, h, dr):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(d1+d2+d3, h), nn.ReLU(), nn.Dropout(p=dr))
            def forward(self, v1, v2, v3):
                return self.encoder(torch.cat([v1, v2, v3], dim=1))
        fusion = TripleConcat(dim1, dim2, dim3, mmhid, dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion



############
# Omic Model
############
class MaxNet(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, dropout_rate=0.25, act=None, label_dim=1, init_max=True):
        super(MaxNet, self).__init__()
        hidden = [64, 48, 32, 32]
        self.act = act

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max: init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.encoder(x)
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False

class GenosTransformer(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, dropout_rate=0.25, act=None, label_dim=1):
        super(GenosTransformer, self).__init__()
        self.act = act
        # Project tabular data to a higher-dimensional embedding space
        self.embedding = nn.Linear(input_dim, 128)
        
        # Transformer Encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, 
            nhead=4, 
            dim_feedforward=512, 
            dropout=dropout_rate, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Output projection to latent mmhid space
        self.post_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, omic_dim)
        )
        
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        # Treat input as a sequence of length 1 for vanilla transformer processing on projected features
        x = self.embedding(x).unsqueeze(1) # [Batch, 1, 128]
        x = self.transformer(x)
        x = x.squeeze(1) # [Batch, 128]
        
        features = self.post_attention(x)
        out = self.classifier(features)
        
        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift
        return features, out

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False

class DisentangleBlock(nn.Module):
    def __init__(self, in_dim, shared_dim, spec_dim):
        super(DisentangleBlock, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(shared_dim)
        )
        self.spec = nn.Sequential(
            nn.Linear(in_dim, spec_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(spec_dim)
        )

    def forward(self, x):
        h_shared = self.shared(x)
        h_spec = self.spec(x)
        return h_shared, h_spec

############
# Graph Model
############
class NormalizeFeaturesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        data.x = data.x.type(torch.cuda.FloatTensor)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class NormalizeEdgesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr.type(torch.cuda.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]#.type(torch.cuda.FloatTensor)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class GraphNet(torch.nn.Module):
    def __init__(self, features=1036, nhid=128, grph_dim=32, nonlinearity=torch.tanh, 
        dropout_rate=0.25, GNN='GCN', use_edges=0, pooling_ratio=0.20, act=None, label_dim=1, init_max=True):
        super(GraphNet, self).__init__()

        # Map GNN string to actual PyG convolution class for SAGPooling
        if GNN == 'GCN':
            GNN = GCNConv
        elif GNN == 'GAT':
            GNN = GATConv
        elif GNN == 'GraphConv':
            GNN = GraphConv
        elif GNN == 'SAGE':
            GNN = SAGEConv

        self.dropout_rate = dropout_rate
        self.use_edges = use_edges
        self.act = act

        self.conv1 = SAGEConv(features, nhid)
        self.pool1 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GNN)#, nonlinearity=nonlinearity)
        self.conv2 = SAGEConv(nhid, nhid)
        self.pool2 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GNN)#, nonlinearity=nonlinearity)
        self.conv3 = SAGEConv(nhid, nhid)
        self.pool3 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GNN)#, nonlinearity=nonlinearity)

        self.lin1 = torch.nn.Linear(nhid*2, nhid)
        self.lin2 = torch.nn.Linear(nhid, grph_dim)
        self.lin3 = torch.nn.Linear(grph_dim, label_dim)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        if init_max: 
            init_max_weights(self)
            print("Initialzing with Max")

    def forward(self, **kwargs):
        data = kwargs['x_grph']
        data = NormalizeFeaturesV2()(data)
        data = NormalizeEdgesV2()(data)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        #x, edge_index, edge_attr, batch = data.x.type(torch.cuda.FloatTensor), data.edge_index.type(torch.cuda.LongTensor), data.edge_attr.type(torch.cuda.FloatTensor), data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        features = F.relu(self.lin2(x))
        out = self.lin3(features)
        if self.act is not None:
            out = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False

############
# Path Model
############
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class PathNet(nn.Module):

    def __init__(self, features, path_dim=32, act=None, num_classes=1, input_size=512*7*7):
        super(PathNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )

        self.linear = nn.Linear(path_dim, num_classes)
        self.act = act

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        if self.features is not None:
            dfs_freeze(self.features)

    def forward(self, **kwargs):
        x = kwargs['x_path']
        # If input is already a feature vector (e.g. [B, 1024] or [B, 512*7*7]), skip extraction
        if x.dim() > 2:
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        
        features = self.classifier(x)
        hazard = self.linear(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False

class UNINet(nn.Module):
    def __init__(self, path_dim=32, act=None, num_classes=1, pretrained=True, input_size=1024):
        super(UNINet, self).__init__()
        import timm
        try:
            # Skip creating the heavy model if we are using pre-extracted features
            if pretrained:
                self.features = timm.create_model('hf-hub:MahmoodLab/uni', pretrained=pretrained, mlp=False)
            else:
                self.features = nn.Identity()
        except Exception as e:
            print(f"Warning: Could not create UNI model via timm. Error: {e}")
            self.features = nn.Identity() 
            
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )
        self.linear = nn.Linear(path_dim, num_classes)
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_path']
        # If input is already a feature vector (B, 1024), bypass feature extraction
        if x.dim() > 2:
            x = self.features(x)
            if isinstance(x, (list, tuple)):
                x = x[0]
            if x.dim() > 2: # handle spatial features if not pooled
                x = torch.mean(x, dim=[2,3])
        
        features = self.classifier(x)
        hazard = self.linear(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return features, hazard



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def get_vgg(arch='vgg19_bn', cfg='E', act=None, batch_norm=True, label_dim=1, pretrained=True, progress=True, **kwargs):
    model = PathNet(make_layers(cfgs[cfg], batch_norm=batch_norm), act=act, num_classes=label_dim, **kwargs)
    
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        for key in list(pretrained_dict.keys()):
            if 'classifier' in key: pretrained_dict.pop(key)

        model.load_state_dict(pretrained_dict, strict=False)
        print("Initializing Path Weights")

    return model



##############################################################################
# Graph + Omic
##############################################################################
class GraphomicNet(nn.Module):
    def __init__(self, opt, act, k):
        super(GraphomicNet, self).__init__()
        self.grph_net = GraphNet(grph_dim=opt.grph_dim, dropout_rate=opt.dropout_rate, use_edges=1, pooling_ratio=0.20, label_dim=opt.label_dim, init_max=False)
        
        if opt.use_transformer:
            self.omic_net = GenosTransformer(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim)
        else:
            self.omic_net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=False)

        if k is not None:
            pt_fname = '_%d.pt' % k
            def find_ckpt(modality, pt_fname):
                possible_paths = [
                    os.path.join(opt.checkpoints_dir, opt.exp_name, modality, modality + pt_fname),
                    os.path.join(opt.checkpoints_dir, opt.exp_name, 'CNN_'+modality.upper(), 'CNN_'+modality.upper() + pt_fname),
                    os.path.join(opt.checkpoints_dir, opt.exp_name, modality, 'CNN_' + modality.upper() + pt_fname)
                ]
                for p in possible_paths:
                    if os.path.exists(p): return p
                return None

            grph_path = find_ckpt('graph', pt_fname)
            omic_path = find_ckpt('omic', pt_fname)

            if grph_path:
                try:
                    best_grph_ckpt = torch.load(grph_path, map_location=torch.device('cpu'), weights_only=False)
                    sd = best_grph_ckpt['model_state_dict']
                    if any(k.startswith('grph_net.') for k in sd.keys()):
                        sd = {k[len('grph_net.'):]: v for k, v in sd.items() if k.startswith('grph_net.')}
                    self.grph_net.load_state_dict(sd, strict=False)
                    print("Loading Graph Model:\n", grph_path)
                except Exception as e:
                    print(f"Warning: Could not load Graph weights from {grph_path}. Error: {e}")

            if omic_path:
                try:
                    best_omic_ckpt = torch.load(omic_path, map_location=torch.device('cpu'), weights_only=False)
                    sd = best_omic_ckpt['model_state_dict']
                    if any(k.startswith('omic_net.') for k in sd.keys()):
                        sd = {k[len('omic_net.'):]: v for k, v in sd.items() if k.startswith('omic_net.')}
                    self.omic_net.load_state_dict(sd, strict=False)
                    print("Loading Omic Model:\n", omic_path)
                except Exception as e:
                    print(f"Warning: Could not load Omic weights from {omic_path}. Error: {e}")

            if not grph_path or not omic_path:
                print("Warning: Missing pre-trained weights for Graph or Omic.")

        if opt.use_disentanglement:
            self.shared_dim = 16
            self.spec_dim = 16
            self.disentangle_grph = DisentangleBlock(opt.grph_dim, self.shared_dim, self.spec_dim)
            self.disentangle_omic = DisentangleBlock(opt.omic_dim, self.shared_dim, self.spec_dim)
            self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.grph_gate, gate2=opt.omic_gate, dim1=self.shared_dim, dim2=self.shared_dim, scale_dim1=opt.grph_scale, scale_dim2=opt.omic_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
            self.classifier = nn.Sequential(nn.Linear(opt.mmhid + self.spec_dim*2, opt.label_dim))
        else:
            self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.grph_gate, gate2=opt.omic_gate, dim1=opt.grph_dim, dim2=opt.omic_dim, scale_dim1=opt.grph_scale, scale_dim2=opt.omic_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
            self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
            
        self.act = act

        dfs_freeze(self.grph_net)
        if not opt.use_transformer:
            dfs_freeze(self.omic_net)
            
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        grph_vec, _ = self.grph_net(x_grph=kwargs['x_grph'])
        omic_vec, _ = self.omic_net(x_omic=kwargs['x_omic'])
        
        if hasattr(self, 'disentangle_grph'):
            h_grph_shared, h_grph_spec = self.disentangle_grph(grph_vec)
            h_omic_shared, h_omic_spec = self.disentangle_omic(omic_vec)
            features = self.fusion(h_grph_shared, h_omic_shared)
            features = torch.cat([features, h_grph_spec, h_omic_spec], dim=1)
        else:
            features = self.fusion(grph_vec, omic_vec)
            
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False


##############################################################################
# Path + Omic
##############################################################################
class PathomicNet(nn.Module):
    def __init__(self, opt, act, k):
        super(PathomicNet, self).__init__()
        if opt.use_uni:
            self.path_net = UNINet(path_dim=opt.path_dim, act=act, num_classes=opt.label_dim)
        else:
            self.path_net = get_vgg(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim)
        
        if opt.use_transformer:
            self.omic_net = GenosTransformer(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim)
        else:
            self.omic_net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=False)

        if k is not None:
            pt_fname = '_%d.pt' % k
            def find_ckpt(modality, pt_fname):
                possible_paths = [
                    os.path.join(opt.checkpoints_dir, opt.exp_name, modality, modality + pt_fname),
                    os.path.join(opt.checkpoints_dir, opt.exp_name, 'CNN_'+modality.upper(), 'CNN_'+modality.upper() + pt_fname),
                    os.path.join(opt.checkpoints_dir, opt.exp_name, modality, 'CNN_' + modality.upper() + pt_fname)
                ]
                for p in possible_paths:
                    if os.path.exists(p): return p
                return None

            path_path = find_ckpt('path', pt_fname)
            omic_path = find_ckpt('omic', pt_fname)

            if path_path:
                try:
                    best_path_ckpt = torch.load(path_path, map_location=torch.device('cpu'), weights_only=False)
                    sd = best_path_ckpt['model_state_dict']
                    if any(k.startswith('path_net.') for k in sd.keys()):
                        sd = {k[len('path_net.'):]: v for k, v in sd.items() if k.startswith('path_net.')}
                    if opt.use_uni:
                        self.path_net.load_state_dict(sd, strict=False)
                    else:
                        pretrained_dict = {k: v for k, v in sd.items() if 'classifier' not in k}
                        self.path_net.load_state_dict(pretrained_dict, strict=False)
                    print("Loading Path Model:\n", path_path)
                except Exception as e:
                    print(f"Warning: Could not load Path weights from {path_path}. Error: {e}")
            
            if omic_path:
                try:
                    best_omic_ckpt = torch.load(omic_path, map_location=torch.device('cpu'), weights_only=False)
                    sd = best_omic_ckpt['model_state_dict']
                    if any(k.startswith('omic_net.') for k in sd.keys()):
                        sd = {k[len('omic_net.'):]: v for k, v in sd.items() if k.startswith('omic_net.')}
                    self.omic_net.load_state_dict(sd, strict=False)
                    print("Loading Omic Model:\n", omic_path)
                except Exception as e:
                    print(f"Warning: Could not load Omic weights from {omic_path}. Error: {e}")
            
            if not path_path or not omic_path:
                print("Warning: Missing pre-trained weights. Check if unimodal training was completed.")

        if opt.use_disentanglement:
            self.shared_dim = 16
            self.spec_dim = 16
            self.disentangle_path = DisentangleBlock(opt.path_dim, self.shared_dim, self.spec_dim)
            self.disentangle_omic = DisentangleBlock(opt.omic_dim, self.shared_dim, self.spec_dim)
            self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.path_gate, gate2=opt.omic_gate, dim1=self.shared_dim, dim2=self.shared_dim, scale_dim1=opt.path_scale, scale_dim2=opt.omic_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
            self.classifier = nn.Sequential(nn.Linear(opt.mmhid + self.spec_dim*2, opt.label_dim))
        else:
            self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.path_gate, gate2=opt.omic_gate, dim1=opt.path_dim, dim2=opt.omic_dim, scale_dim1=opt.path_scale, scale_dim2=opt.omic_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
            self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
            
        self.act = act

        if not opt.use_uni and not opt.unfreeze_path: # Freeze VGG features unless unfreeze_path is set
            dfs_freeze(self.path_net.features)
        
        # Don't freeze omic if using transformer (we want to train it)
        if not opt.use_transformer:
            dfs_freeze(self.omic_net)
            
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec, _ = self.path_net(x_path=kwargs['x_path'])
        omic_vec, _ = self.omic_net(x_omic=kwargs['x_omic'])
        
        if hasattr(self, 'disentangle_path'):
            h_path_shared, h_path_spec = self.disentangle_path(path_vec)
            h_omic_shared, h_omic_spec = self.disentangle_omic(omic_vec)
            features = self.fusion(h_path_shared, h_omic_shared)
            features = torch.cat([features, h_path_spec, h_omic_spec], dim=1)
        else:
            features = self.fusion(path_vec, omic_vec)
            
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False



#############################################################################
# Path + Graph + Omic
##############################################################################
class PathgraphomicNet(nn.Module):
    def __init__(self, opt, act, k):
        super(PathgraphomicNet, self).__init__()
        if opt.use_uni:
            self.path_net = UNINet(path_dim=opt.path_dim, act=act, num_classes=opt.label_dim)
        else:
            self.path_net = get_vgg(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim)

        self.grph_net = GraphNet(grph_dim=opt.grph_dim, dropout_rate=opt.dropout_rate, use_edges=1, pooling_ratio=0.20, label_dim=opt.label_dim, init_max=False)
        
        if opt.use_transformer:
            self.omic_net = GenosTransformer(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim)
        else:
            self.omic_net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=False)

        if k is not None:
            pt_fname = '_%d.pt' % k
            def find_ckpt(modality, pt_fname):
                possible_paths = [
                    os.path.join(opt.checkpoints_dir, opt.exp_name, modality, modality + pt_fname),
                    os.path.join(opt.checkpoints_dir, opt.exp_name, 'CNN_'+modality.upper(), 'CNN_'+modality.upper() + pt_fname),
                    os.path.join(opt.checkpoints_dir, opt.exp_name, modality, 'CNN_' + modality.upper() + pt_fname)
                ]
                for p in possible_paths:
                    if os.path.exists(p): return p
                return None

            path_path = find_ckpt('path', pt_fname)
            grph_path = find_ckpt('graph', pt_fname)
            omic_path = find_ckpt('omic', pt_fname)

            if path_path:
                try:
                    best_path_ckpt = torch.load(path_path, map_location=torch.device('cpu'), weights_only=False)
                    sd = best_path_ckpt['model_state_dict']
                    if any(k.startswith('path_net.') for k in sd.keys()):
                        sd = {k[len('path_net.'):]: v for k, v in sd.items() if k.startswith('path_net.')}
                    if opt.use_uni:
                        self.path_net.load_state_dict(sd, strict=False)
                    else:
                        pretrained_dict = {k: v for k, v in sd.items() if 'classifier' not in k}
                        self.path_net.load_state_dict(pretrained_dict, strict=False)
                    print("Loading Path Model:\n", path_path)
                except Exception as e:
                    print(f"Warning: Could not load Path weights from {path_path}. Error: {e}")

            if grph_path:
                try:
                    best_grph_ckpt = torch.load(grph_path, map_location=torch.device('cpu'), weights_only=False)
                    sd = best_grph_ckpt['model_state_dict']
                    if any(k.startswith('grph_net.') for k in sd.keys()):
                        sd = {k[len('grph_net.'):]: v for k, v in sd.items() if k.startswith('grph_net.')}
                    self.grph_net.load_state_dict(sd, strict=False)
                    print("Loading Graph Model:\n", grph_path)
                except Exception as e:
                    print(f"Warning: Could not load Graph weights from {grph_path}. Error: {e}")

            if omic_path:
                try:
                    best_omic_ckpt = torch.load(omic_path, map_location=torch.device('cpu'), weights_only=False)
                    sd = best_omic_ckpt['model_state_dict']
                    if any(k.startswith('omic_net.') for k in sd.keys()):
                        sd = {k[len('omic_net.'):]: v for k, v in sd.items() if k.startswith('omic_net.')}
                    self.omic_net.load_state_dict(sd, strict=False)
                    print("Loading Omic Model:\n", omic_path)
                except Exception as e:
                    print(f"Warning: Could not load Omic weights from {omic_path}. Error: {e}")

        if opt.use_disentanglement:
            self.shared_dim = 16
            self.spec_dim = 16
            self.disentangle_path = DisentangleBlock(opt.path_dim, self.shared_dim, self.spec_dim)
            self.disentangle_grph = DisentangleBlock(opt.grph_dim, self.shared_dim, self.spec_dim)
            self.disentangle_omic = DisentangleBlock(opt.omic_dim, self.shared_dim, self.spec_dim)
            self.fusion = define_trifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.path_gate, gate2=opt.grph_gate, gate3=opt.omic_gate, dim1=self.shared_dim, dim2=self.shared_dim, dim3=self.shared_dim, scale_dim1=opt.path_scale, scale_dim2=opt.grph_scale, scale_dim3=opt.omic_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
            self.classifier = nn.Sequential(nn.Linear(opt.mmhid + self.spec_dim*3, opt.label_dim))
        else:
            self.fusion = define_trifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.path_gate, gate2=opt.grph_gate, gate3=opt.omic_gate, dim1=opt.path_dim, dim2=opt.grph_dim, dim3=opt.omic_dim, scale_dim1=opt.path_scale, scale_dim2=opt.grph_scale, scale_dim3=opt.omic_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
            self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
            
        self.act = act

        if not opt.use_uni and not opt.unfreeze_path:
            dfs_freeze(self.path_net.features)
        dfs_freeze(self.grph_net)
        if not opt.use_transformer:
            dfs_freeze(self.omic_net)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec, _ = self.path_net(x_path=kwargs['x_path'])
        grph_vec, _ = self.grph_net(x_grph=kwargs['x_grph'])
        omic_vec, _ = self.omic_net(x_omic=kwargs['x_omic'])
        
        if hasattr(self, 'disentangle_path'):
            h_path_shared, h_path_spec = self.disentangle_path(path_vec)
            h_grph_shared, h_grph_spec = self.disentangle_grph(grph_vec)
            h_omic_shared, h_omic_spec = self.disentangle_omic(omic_vec)
            features = self.fusion(h_path_shared, h_grph_shared, h_omic_shared)
            features = torch.cat([features, h_path_spec, h_grph_spec, h_omic_spec], dim=1)
        else:
            features = self.fusion(path_vec, grph_vec, omic_vec)
            
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False



##############################################################################
# Ensembling Effects
##############################################################################
class PathgraphNet(nn.Module):
    def __init__(self, opt, act, k):
        super(PathgraphNet, self).__init__()
        if opt.use_uni:
            self.path_net = UNINet(path_dim=opt.path_dim, act=act, num_classes=opt.label_dim)
        else:
            self.path_net = get_vgg(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim)

        self.grph_net = GraphNet(grph_dim=opt.grph_dim, dropout_rate=opt.dropout_rate, use_edges=1, pooling_ratio=0.20, label_dim=opt.label_dim, init_max=False)

        if k is not None:
            pt_fname = '_%d.pt' % k
            def find_ckpt(modality, pt_fname):
                possible_paths = [
                    os.path.join(opt.checkpoints_dir, opt.exp_name, modality, modality + pt_fname),
                    os.path.join(opt.checkpoints_dir, opt.exp_name, 'CNN_'+modality.upper(), 'CNN_'+modality.upper() + pt_fname),
                    os.path.join(opt.checkpoints_dir, opt.exp_name, modality, 'CNN_' + modality.upper() + pt_fname)
                ]
                for p in possible_paths:
                    if os.path.exists(p): return p
                return None

            path_path = find_ckpt('path', pt_fname)
            grph_path = find_ckpt('graph', pt_fname)
            
            if path_path:
                best_path_ckpt = torch.load(path_path, map_location=torch.device('cpu'), weights_only=False)
                if opt.use_uni:
                    self.path_net.load_state_dict(best_path_ckpt['model_state_dict'])
                else:
                    pretrained_dict = {k: v for k, v in best_path_ckpt['model_state_dict'].items() if 'classifier' not in k}
                    self.path_net.load_state_dict(pretrained_dict, strict=False)
                print("Loading Path Model:\n", path_path)

            if grph_path:
                best_grph_ckpt = torch.load(grph_path, map_location=torch.device('cpu'), weights_only=False)
                self.grph_net.load_state_dict(best_grph_ckpt['model_state_dict'])
                print("Loading Graph Model:\n", grph_path)
            
            if not path_path or not grph_path:
                print("Warning: Missing pre-trained weights for PathgraphNet.")

        self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.path_gate, gate2=opt.grph_gate, dim1=opt.path_dim, dim2=opt.grph_dim, scale_dim1=opt.path_scale, scale_dim2=opt.grph_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act

        if not opt.use_uni and not opt.unfreeze_path:
            dfs_freeze(self.path_net.features)
        dfs_freeze(self.grph_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec, _ = self.path_net(x_path=kwargs['x_path'])
        grph_vec, _ = self.grph_net(x_grph=kwargs['x_grph'])
        features = self.fusion(path_vec, grph_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False


class PathpathNet(nn.Module):
    def __init__(self, opt, act, k):
        super(PathpathNet, self).__init__()
        self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.path_gate, gate2=1-opt.path_gate if opt.path_gate else 0, 
            dim1=opt.path_dim, dim2=opt.path_dim, scale_dim1=opt.path_scale, scale_dim2=opt.path_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec = kwargs['x_path']
        features = self.fusion(path_vec, path_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False


class GraphgraphNet(nn.Module):
    def __init__(self, opt, act, k):
        super(GraphgraphNet, self).__init__()
        self.grph_net = GraphNet(grph_dim=opt.grph_dim, dropout_rate=opt.dropout_rate, use_edges=1, pooling_ratio=0.20, label_dim=opt.label_dim, init_max=False)
        if k is not None:
            pt_fname = '_%d.pt' % k
            best_grph_ckpt = torch.load(os.path.join(opt.checkpoints_dir, opt.exp_name, 'graph', 'graph'+pt_fname), map_location=torch.device('cpu'), weights_only=False)
            self.grph_net.load_state_dict(best_grph_ckpt['model_state_dict'])
            print("Loading Models:\n", os.path.join(opt.checkpoints_dir, opt.exp_name, 'graph', 'graph'+pt_fname))
        self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.grph_gate, gate2=1-opt.grph_gate if opt.grph_gate else 0, 
            dim1=opt.grph_dim, dim2=opt.grph_dim, scale_dim1=opt.grph_scale, scale_dim2=opt.grph_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act
        dfs_freeze(self.grph_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        grph_vec, _ = self.grph_net(x_grph=kwargs['x_grph'])
        features = self.fusion(grph_vec, grph_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False


class OmicomicNet(nn.Module):
    def __init__(self, opt, act, k):
        super(OmicomicNet, self).__init__()
        self.omic_net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=False)
        if k is not None:
            pt_fname = '_%d.pt' % k
            best_omic_ckpt = torch.load(os.path.join(opt.checkpoints_dir, opt.exp_name, 'omic', 'omic'+pt_fname), map_location=torch.device('cpu'), weights_only=False)
            self.omic_net.load_state_dict(best_omic_ckpt['model_state_dict'])
            print("Loading Models:\n", os.path.join(opt.checkpoints_dir, opt.exp_name, 'omic', 'omic'+pt_fname))
        self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.omic_gate, gate2=1-opt.omic_gate if opt.omic_gate else 0, 
            dim1=opt.omic_dim, dim2=opt.omic_dim, scale_dim1=opt.omic_scale, scale_dim2=opt.omic_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act
        dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        omic_vec, _ = self.omic_net(x_omic=kwargs['x_omic'])
        features = self.fusion(omic_vec, omic_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False