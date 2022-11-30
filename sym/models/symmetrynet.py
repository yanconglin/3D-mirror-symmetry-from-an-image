import itertools
import math
import random
import time
import sys
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import torch
import torch.nn as nn
import torch.nn.functional as F

from sym.config import CI, CM
from sym.models.backbone import FeatureNet
from sym.models.sphere import IMG2SPHERE
from sym.models.pairconv import PairConv
from sym.models.dgcn import SimpleDGCN
from sym.models.loss import Loss_pos_neg
from itertools import accumulate

class SymmetryNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = FeatureNet()
        self.conv_add = nn.Sequential(
            nn.Conv2d(64, CM.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CM.C),
            nn.ReLU(inplace=True),
        )
        """check IMG2SPHERE for detailed explanation of the hyper parameter 'side_flag' """
        self.img2sphere = IMG2SPHERE(CM.D, CM.depth_min, CM.depth_max, side_flag=1e16)
        self.pairconv = PairConv(C_in=CM.D, C_out=16)
        self.dgcn1 = SimpleDGCN(nf=[CM.C_sphere, CM.C_sphere, CM.C_sphere],
                               num_nodes=CM.num_nodes[0],
                               num_neighbors=CM.num_neighbors[0])

        self.dgcn2 = SimpleDGCN(nf=[CM.C_sphere, CM.C_sphere, CM.C_sphere],
                               num_nodes=CM.num_nodes[1],
                               num_neighbors=CM.num_neighbors[1])

        self.dgcn3 = SimpleDGCN(nf=[CM.C_sphere, CM.C_sphere, CM.C_sphere],
                               num_nodes=CM.num_nodes[2],
                               num_neighbors=CM.num_neighbors[2])

        self.loss = Loss_pos_neg()

        # # # DGCN configuration
        self.num_nodes = CM.num_nodes
        self.num_neighbors = list(CM.num_neighbors)
        num_edges = [CM.num_nodes[l] * CM.num_neighbors[l] for l in range(CM.n_levels)]
        self.num_edges = num_edges
        print('dgcn num_nodes/num_neighbors/num_edges', self.num_nodes, self.num_neighbors, self.num_edges)
        self.nodes = list(accumulate(CM.num_nodes))
        self.edges = list(accumulate(num_edges))
        print('dgcn nodes/edges', self.nodes, self.edges)

    def forward(self, input_dict, mode='train/valid'):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        x = self.backbone(input_dict["image"])
        x = self.conv_add(x)

        B, C, H, W = x.shape
        x_t = x.view(B, C, H * W)
        corr = torch.bmm(x_t.transpose(1, 2), x_t) / (C ** 0.5)
            
        if mode != 'test':
            corr = self.img2sphere(corr, input_dict["Ss"], input_dict["Ps"])
            # print('im2sphere', corr.shape)
            sphere = self.pairconv(corr)
            
            ########## level 1 #####################################
            sphere1 = self.dgcn1(sphere[:, 0:self.nodes[0]], input_dict["edge_index"][:, :, 0:self.edges[0]])
            sphere1 = sphere1.view(B, -1, 1).softmax(dim=1).view(-1, 1)
            ########## level 2 #####################################
            sphere2 = self.dgcn2(sphere[:, self.nodes[0]:self.nodes[1]], input_dict["edge_index"][:, :, self.edges[0]:self.edges[1]])
            sphere2 = sphere2.view(B, -1, 1).softmax(dim=1).view(-1, 1)
            ########## level 3 #####################################
            sphere3 = self.dgcn3(sphere[:, self.nodes[1]:self.nodes[2]], input_dict["edge_index"][:, :, self.edges[1]:self.edges[2]])
            sphere3 = sphere3.view(B, -1, 1).softmax(dim=1).view(-1, 1)

            losses = {}
            loss_pos1, loss_neg1 = self.loss(sphere1, input_dict["label"][:, 0:self.nodes[0]])
            loss_pos2, loss_neg2 = self.loss(sphere2, input_dict["label"][:, self.nodes[0]:self.nodes[1]])
            loss_pos3, loss_neg3 = self.loss(sphere3, input_dict["label"][:, self.nodes[1]:self.nodes[2]])

            losses["loss_pos1"] = loss_pos1 * CM.lpos
            losses["loss_neg1"] = loss_neg1 * CM.lneg
            losses["loss_pos2"] = loss_pos2 * CM.lpos
            losses["loss_neg2"] = loss_neg2 * CM.lneg
            losses["loss_pos3"] = loss_pos3 * CM.lpos
            losses["loss_neg3"] = loss_neg3 * CM.lneg

            return {
                "losses": losses,
                "metrics": {},
                "preds1": sphere1.reshape(-1, CM.num_nodes[0]),
                "preds2": sphere2.reshape(-1, CM.num_nodes[1]),
                "preds3": sphere3.reshape(-1, CM.num_nodes[2]),
            }
        else:
            K = input_dict['K'].squeeze(0)
            ########## level 1 #####################################
            pts = input_dict['pts'].squeeze(0)
            corr1 = self.img2sphere(corr, input_dict["Ss"], input_dict["Ps"])
            sphere1 = self.pairconv(corr1)
            sphere1 = self.dgcn1(sphere1, input_dict["edge_index"])
            argmax = torch.argmax(sphere1.flatten())
            best_w = pts[argmax]


            ########## level 2 #####################################
            pts, Ss, Ps, edge_index = compute_graph(best_w, CM.theta[1], CM.num_nodes[1], CM.num_neighbors[1], K)
            corr2 = self.img2sphere(corr, Ss[None], Ps[None])
            sphere2 = self.pairconv(corr2)
            sphere2 = self.dgcn2(sphere2, edge_index[None])
            argmax = torch.argmax(sphere2.flatten())
            best_w = pts[argmax]
            
            ########## level 3 #####################################
            pts, Ss, Ps, edge_index = compute_graph(best_w, CM.theta[2], CM.num_nodes[2], CM.num_neighbors[2], K)
            corr3 = self.img2sphere(corr, Ss[None], Ps[None])
            sphere3 = self.pairconv(corr3)
            sphere3 = self.dgcn3(sphere3, edge_index[None])
            argmax = torch.argmax(sphere3.flatten())
            best_w = pts[argmax]
            
            # end.record()
            # torch.cuda.synchronize()
            # cur_time = start.elapsed_time(end)*1e-3
            # print(f'########### cur_time time/frame:{cur_time:.4f}, FPS:{1.0 / (cur_time):.4f} ######################')
            return {
                "metrics": {},
                "best_w": best_w,
            }


def compute_graph(best_w, theta, num_nodes, num_neighbors, K):
    pts = gold_spiral_sampling_patch(best_w / torch.norm(best_w), theta * math.pi / 180., num_nodes)
    ws = pts / pts[:, 2:3]

    K_inv = torch.inverse(K).repeat(num_nodes, 1, 1)
    Ss = w2S(ws)
    Ss = torch.bmm(K.repeat(num_nodes, 1, 1), Ss)
    Ss = torch.bmm(Ss, K_inv)
    Ps = w2P(ws)
    Ps = torch.bmm(Ps[:, None], K_inv).squeeze(1)

    dist_cos = pts @ pts.t()
    _, topk_min_idx = torch.topk(dist_cos.abs(), k=num_neighbors, dim=1)
    neighbors = topk_min_idx.flatten()
    centers = torch.arange(0, num_nodes, dtype=best_w.dtype, device=best_w.device).repeat_interleave(num_neighbors)
    edge_index = torch.stack([centers, neighbors], dim=0).long()
    return pts, Ss, Ps, edge_index

def gold_spiral_sampling_patch(v, alpha, num_pts):
    v1 = orth(v)
    v2 = torch.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = torch.linspace(1, num_pts, num_pts, dtype=v.dtype, device=v.device)
    
    phi = 1 + (math.cos(alpha) - 1) * indices / num_pts
    phi = torch.acos(phi)
    theta = math.pi * (1 + 5 ** 0.5) * indices
    r = torch.sin(phi)
    w = (v * torch.cos(phi) + r * (v1 * torch.cos(theta) + v2 * torch.sin(theta))).T
    return w


def orth(v):
    x, y, z = v  # [3]
    o = v.new_tensor([0.0, -z.item(), y.item()]) if x.abs() < y.abs() else v.new_tensor([-z.item(), 0.0, x.item()])
    o /= torch.norm(o)
    return o


# batched    
def w2S(w):
    m, n = w.shape
    S = torch.eye(4, device=w.device, dtype=w.dtype)[None].repeat(m,1,1)
    S[:, :3, :3] = torch.eye(3, device=w.device, dtype=w.dtype)[None].repeat(m,1,1) - 2 * torch.bmm(w[:,:,None], w[:, None,:]) / torch.sum(w ** 2, dim=1)[:, None, None]
    S[:, :3, 3] = -2 * w / torch.sum(w ** 2, dim=1)[:, None]
    return S

# batched 
def w2P(w):
    m, n = w.shape
    P = w.new_zeros((m, 4))
    P[:,:3] = w
    P[:, 3] = 1.0
    return P / torch.norm(w, dim=1, keepdim=True)


