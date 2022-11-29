import os
import numpy as np
import itertools
import torch
import torch.nn as nn

class Loss_pos_neg(nn.Module):
    def __init__(self):
        super(Loss_pos_neg, self).__init__()
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, output, label):
        label = label.flatten().view(-1,1)
        # print('loss', output.shape, label.shape)
        ### separate the pos and neg loss values
        loss = self.loss(output, label)
        loss_pos = loss[label>0.0].sum().float() / label.gt(0.0).sum().float()
        loss_neg = loss[label==0.0].sum().float() / (label.nelement() - label.gt(0.0).sum().float())
        # print('loss_pos, loss_neg', loss_pos.shape, loss_neg.shape)
        # # # in mutli-gpus case, better to make tensors at least one-dim to stack, otherwise there is a warning
        return loss_pos.unsqueeze(0), loss_neg.unsqueeze(0)

