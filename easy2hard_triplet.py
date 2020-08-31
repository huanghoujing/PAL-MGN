#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import norm
import numpy as np

class TripletSemihardLoss(nn.Module):
    """
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self, margin=0, DEVICE = '0', size_average=True, sampling = 'batch_hard', batch_id=16, batch_image=8):
        super(TripletSemihardLoss, self).__init__()
        self.DEVICE = DEVICE;
        self.margin = margin
        self.size_average = size_average
        self.sampling = sampling
        self.batch_id = batch_id
        self.batch_image = batch_image
            
    def forward(self, input, target, epoch):
        y_true = target.int().unsqueeze(-1)
        same_id = torch.eq(y_true, y_true.t()).type_as(input)

        pos_mask = same_id
        neg_mask = 1 - same_id

        # output[i, j] = || feature[i, :] - feature[j, :] ||_2
        dist_squared = torch.sum(input ** 2, dim=1, keepdim=True) + \
                       torch.sum(input.t() ** 2, dim=0, keepdim=True) - \
                       2.0 * torch.matmul(input, input.t())
        dist = dist_squared.clamp(min=1e-16).sqrt()
        
        def _mask_max(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor - 1e6 * (1 - mask)
            _max, _idx = torch.max(input_tensor, dim=axis, keepdim=keepdims)
            return _max, _idx

        def _mask_min(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor + 1e6 * (1 - mask)
            _min, _idx = torch.min(input_tensor, dim=axis, keepdim=keepdims)
            return _min, _idx
            
        if(self.sampling == 'batch_hard'):
            pos_max, pos_idx = _mask_max(dist, pos_mask, axis=-1)
            neg_min, neg_idx = _mask_min(dist, neg_mask, axis=-1)
            y = torch.ones(same_id.size()[0]).to(self.DEVICE)
        if(self.sampling == 'curriculum'):
            pos_max = []
            neg_min = []
            t0 = 20.0
            t1 = 40.0
            
            Num_neg = self.batch_id*self.batch_image-self.batch_image
            mu = max(Num_neg-Num_neg/t0*epoch, 0.0)
            sigma = 15*0.001**(max((epoch-t0)/(t1-t0), 0.0))
            neg_probs = norm(mu, sigma).pdf(np.linspace(0,Num_neg-1,Num_neg))
            neg_probs = torch.from_numpy(neg_probs).clamp(min=3e-5, max=20.0)
            for i in range(self.batch_id):
                for j in range(self.batch_image):
                    neg_examples = dist[i*self.batch_image+j][neg_mask[i*self.batch_image+j] == 1]
                    sort_neg_examples = torch.sort(neg_examples)[0]
                    for pair in range(j+1,self.batch_image):
                        pos_max.append(dist[i*self.batch_image+j][i*self.batch_image+pair].unsqueeze(dim=0))
                        choosen_neg = sort_neg_examples[torch.multinomial(neg_probs,1).to(self.DEVICE)]
                        neg_min.append(choosen_neg.unsqueeze(dim=0))  
            
            pos_max = torch.cat(pos_max).to(self.DEVICE) 
            neg_min = torch.cat(neg_min).to(self.DEVICE) 
            y = torch.ones(pos_max.size()).to(self.DEVICE)
                   
          
        return F.margin_ranking_loss(neg_min.float(),
                                     pos_max.float(),
                                     y,
                                     self.margin,
                                     self.size_average)
