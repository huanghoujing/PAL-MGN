from __future__ import print_function

import sys
import argparse
import copy
import os
import os.path as osp
import time
import datetime

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet50, Bottleneck
from torchvision.transforms import functional
import torch.nn.functional as F

from __init__ import cmc, mean_ap
from market1501 import Market1501, RandomIdSampler
from msmt17 import MSMT17
from partial_reid import PartialREID
from partial_ilids import PartialiLIDs
from easy2hard_triplet import TripletSemihardLoss
from random_erasing import RandomErasing
import shutil
from pa_pool import pa_max_pool
from ps_head import PartSegHead, PartSegHeadConv
from ps_loss import PSLoss
from np_distance import compute_dist_with_visibility
from file_utils import load_pickle, save_pickle


class MGN(nn.Module):
    def __init__(self, num_classes, args):
        super(MGN, self).__init__()

        self.args = args
        resnet = resnet50(pretrained=False)
        res_path = os.path.dirname(os.path.realpath(__file__)) + '/resnet50-19c8e357.pth'
        resnet.load_state_dict(torch.load(res_path))

        # backbone
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3[0]# res_conv4_1
        )

        # res_conv4x
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        # res_conv5 global
        res_g_conv5 = resnet.layer4
        # res_conv5 part
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        # mgn part-1 global
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5 if args.head_1part_stride == 2 else res_p_conv5))
        # mgn part-2
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        # mgn part-3
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        # global max pooling
        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4) if args.head_1part_stride == 2 else (24, 8))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))

        add_part_2048 = nn.Sequential(nn.BatchNorm1d(2048), nn.ReLU())
        self._init_add_part(add_part_2048)
        self.add_part_1 = copy.deepcopy(add_part_2048)
        self.add_part_2 = copy.deepcopy(add_part_2048)
        self.add_part_3 = copy.deepcopy(add_part_2048)

        
        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # fc softmax loss
        self.fc_id_2048_0_tmp = nn.Linear(2048, 2048)
        self.fc_id_2048_1_tmp = nn.Linear(2048, 2048)
        self.fc_id_2048_2_tmp = nn.Linear(2048, 2048)
        self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_1 = nn.Linear(2048, num_classes)
        self.fc_id_2048_2 = nn.Linear(2048, num_classes)
        self.fc_id_256_1_0 = nn.Linear(256, num_classes)
        self.fc_id_256_1_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_0 = nn.Linear(256, num_classes)
        self.fc_id_256_2_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_2 = nn.Linear(256, num_classes)

        self._init_fc(self.fc_id_2048_0_tmp)
        self._init_fc(self.fc_id_2048_1_tmp)
        self._init_fc(self.fc_id_2048_2_tmp)
        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)
        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

        embedding = nn.Sequential(nn.Linear(256, 256))
        self.embedding_1 = copy.deepcopy(embedding)
        self.embedding_2 = copy.deepcopy(embedding)
        self.embedding_3 = copy.deepcopy(embedding)
        self._init_embedding(self.embedding_1)
        self._init_embedding(self.embedding_2)
        self._init_embedding(self.embedding_3)

        if args.src_ps_lw > 0 or args.cd_ps_lw > 0:
            ps_head_cls = PartSegHead
            self.ps_head = ps_head_cls({'in_c': 2048, 'mid_c': 256, 'num_classes': 8})
        print('Model Structure:')
        print(self)
        
    @staticmethod
    def _init_embedding(embedding):
        nn.init.normal_(embedding[0].weight, std=0.01)
        nn.init.constant_(embedding[0].bias, 0.)
        
    @staticmethod
    def _init_add_part(add_part):
        nn.init.normal_(add_part[0].weight, mean = 1.0, std=0.02)
        nn.init.constant_(add_part[0].bias, 0.)
        
    @staticmethod
    def _init_reduction(reduction):
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        nn.init.normal_(reduction[1].weight, mean = 1.0, std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)     

    @staticmethod
    def _init_fc(fc):
        nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, in_dict):
        x = self.backbone(in_dict['im'])

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        if hasattr(self, 'ps_head'):
            ps1 = self.ps_head(p1)
            ps2 = self.ps_head(p2)
            ps3 = self.ps_head(p3)

        zg_p1 = self.maxpool_zg_p1(p1)  # z_g^G
        zg_p2 = self.maxpool_zg_p2(p2)  # z_g^P2
        zg_p3 = self.maxpool_zg_p3(p3)  # z_g^P3

        if args.pap:
            pap_pooled = pa_max_pool({'feat': p2, 'pap_mask': in_dict['pap_mask_2p']})
            z0_p2, z1_p2 = pap_pooled['feat_list']
            part_2_1_v, part_2_2_v = pap_pooled['visible'][:, 0], pap_pooled['visible'][:, 1]
        else:
            zp2 = F.max_pool2d(p2, (12, 8))
            z0_p2 = zp2[:, :, 0:1, :]  # z_p0^P2
            z1_p2 = zp2[:, :, 1:2, :]  # z_p1^P2

        if args.pap:
            pap_pooled = pa_max_pool({'feat': p3, 'pap_mask': in_dict['pap_mask_3p']})
            z0_p3, z1_p3, z2_p3 = pap_pooled['feat_list']
            part_3_1_v, part_3_2_v, part_3_3_v = pap_pooled['visible'][:, 0], pap_pooled['visible'][:, 1], pap_pooled['visible'][:, 2]
        else:
            zp3 = F.max_pool2d(p3, (8, 8))
            z0_p3 = zp3[:, :, 0:1, :]  # z_p0^P3
            z1_p3 = zp3[:, :, 1:2, :]  # z_p1^P3
            z2_p3 = zp3[:, :, 2:3, :]  # z_p2^P3
        
        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)  # f_g^G, L_triplet^G
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)  # f_g^P2, L_triplet^P2
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)  # f_g^P3, L_triplet^P3
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)  # f_p0^P2
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)  # f_p1^P2
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)  # f_p0^P3
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)  # f_p1^P3
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)  # f_p2^P3
        
        fg_p1 = self.embedding_1(fg_p1)
        fg_p2 = self.embedding_2(fg_p2)
        fg_p3 = self.embedding_3(fg_p3)

        l_p1 = self.fc_id_2048_0_tmp(zg_p1.squeeze(dim=3).squeeze(dim=2))  # L_softmax^G
        l_p2 = self.fc_id_2048_1_tmp(zg_p2.squeeze(dim=3).squeeze(dim=2))  # L_softmax^P2
        l_p3 = self.fc_id_2048_2_tmp(zg_p3.squeeze(dim=3).squeeze(dim=2))  # L_softmax^P3
        
        l_p1 = self.add_part_1(l_p1)
        l_p2 = self.add_part_2(l_p2)
        l_p3 = self.add_part_3(l_p3)

        l_p1 = self.fc_id_2048_0(l_p1)  # L_softmax^G
        l_p2 = self.fc_id_2048_1(l_p2)  # L_softmax^P2
        l_p3 = self.fc_id_2048_2(l_p3)  # L_softmax^P3

        l0_p2 = self.fc_id_256_1_0(f0_p2)  # L_softmax0^P2
        l1_p2 = self.fc_id_256_1_1(f1_p2)  # L_softmax1^P2
        l0_p3 = self.fc_id_256_2_0(f0_p3)  # L_softmax0^P3
        l1_p3 = self.fc_id_256_2_1(f1_p3)  # L_softmax1^P3
        l2_p3 = self.fc_id_256_2_2(f2_p3)  # L_softmax2^P3
      
        predict_1 = torch.cat([0.8*f0_p2, f1_p2, 0.7*f0_p3, f1_p3, 0.7*f2_p3], dim=1)
        predict_2 = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1) #67575
        if hasattr(self, 'ps_head') and args.pap:
            return predict_1, predict_2, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3, part_2_1_v, part_2_2_v, part_3_1_v, part_3_2_v, part_3_3_v, ps1, ps2, ps3
        elif hasattr(self, 'ps_head') and not args.pap:
            return predict_1, predict_2, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3, ps1, ps2, ps3
        elif not hasattr(self, 'ps_head') and args.pap:
            return predict_1, predict_2, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3, part_2_1_v, part_2_2_v, part_3_1_v, part_3_2_v, part_3_3_v
        else:
            return predict_1, predict_2, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3

def save_model(model, filename):
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    for key in state: 
        state[key] = state[key].clone().cpu()
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)

def load_model_weight(model, model_weight_file):
    assert osp.exists(model_weight_file), "model_weight_file {} does not exist!".format(model_weight_file)
    assert osp.isfile(model_weight_file), "model_weight_file {} is not file!".format(model_weight_file)
    model_weight = torch.load(model_weight_file, map_location=(lambda storage, loc: storage))
    model.load_state_dict(model_weight)
    msg = '=> Loaded model_weight from {}'.format(model_weight_file)
    print(msg)

def get_dataset_root(name):
    if name == 'market1501':
        root = 'Market-1501-v15.09.15'
    elif name == 'cuhk03':
        root = 'cuhk03-np-jpg/detected'
    elif name == 'duke':
        root = 'DukeMTMC-reID'
    else:
        raise ValueError
    return root


class InfiniteNextBatch(object):
    def __init__(self, loader):
        self.loader = loader
        self.reset()

    def reset(self):
        self.loader_iter = iter(self.loader)

    def next_batch(self):
        try:
            batch = self.loader_iter.next()
        except StopIteration:
            self.reset()
            batch = self.loader_iter.next()
        return batch


def get_next_batch(loader):
    try:
        batch = loader.next()
    except StopIteration:
        batch = loader.next()
    return batch


def run(args):
    gpuId, epochs, weight_decay, batch_id, batch_image, lr_1, lr_2, erasing_p, sampling, exp_dir, trainset_name, cd_trainset_name, testset_names, rand_crop, head_1part_stride = \
        args.gpuId, args.epochs, args.weight_decay, args.batch_id, args.batch_image, args.lr_1, args.lr_2, args.erasing_p, args.sampling, args.exp_dir, args.trainset_name, args.cd_trainset_name, args.testset_names, args.rand_crop, args.head_1part_stride

    DEVICE = torch.device("cuda:" + gpuId if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    num_workers = 4

    batch_test = 64 #32

    train_list = [transforms.Resize((400, 144)), transforms.RandomCrop((384, 128))] if rand_crop else [transforms.Resize((384, 128))]
    train_list += [
        transforms.ToTensor(),
    ]
    if erasing_p>0:
        train_list = train_list +  [RandomErasing(probability = erasing_p, mean =[0.0, 0.0, 0.0])]
    train_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    
    train_transform = transforms.Compose(train_list)
    if trainset_name in ['market1501', 'cuhk03', 'duke']:
        root = get_dataset_root(trainset_name)
        train_dataset = Market1501(root + '/bounding_box_train', transform=train_transform, training=True, kpt_file=trainset_name+'-kpt.pkl' if args.pap else None, ps_dir=root+'_ps_label' if args.src_ps_lw > 0 else None)
    elif trainset_name in ['msmt17']:
        train_dataset = MSMT17(transform=train_transform, training=True, use_kpt=args.pap, use_ps=args.src_ps_lw > 0, split='train')
    else:
        raise ValueError('Invalid train set {}'.format(trainset_name))
    train_loader = DataLoader(train_dataset,
                                         sampler=RandomIdSampler(train_dataset, batch_image=batch_image),
                                         batch_size=batch_id * batch_image,
                                         num_workers=num_workers, drop_last=True)
                                         
    if args.cd_ps_lw > 0:
        if cd_trainset_name in ['market1501', 'cuhk03', 'duke']:
            cd_train_dataset = Market1501(get_dataset_root(cd_trainset_name) + '/bounding_box_train', transform=train_transform, training=True, kpt_file=None, ps_dir=cd_trainset_name + '-ps')
        elif cd_trainset_name in ['msmt17']:
            cd_train_dataset = MSMT17(transform=train_transform, training=True, use_kpt=False, use_ps=True)
        else:
            raise ValueError('Invalid cd train set {}'.format(cd_trainset_name))
        cd_train_loader = InfiniteNextBatch(DataLoader(cd_train_dataset,
                                  batch_size=args.cd_train_batch_size,
                                  num_workers=num_workers, drop_last=True))

    test_transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_flip_transform = transforms.Compose([
        transforms.Resize((384, 128)),
        functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def make_test_loader_M_C_D(root, name):
        query_dataset = Market1501(root + '/query', transform=test_transform, training=False, kpt_file=name+'-kpt.pkl' if args.pap else None)
        query_flip_dataset = Market1501(root + '/query', transform=test_flip_transform, training=False, kpt_file=name+'-kpt.pkl' if args.pap else None)
        query_loader = DataLoader(query_dataset, batch_size=batch_test, num_workers=num_workers)
        query_flip_loader = DataLoader(query_flip_dataset, batch_size=batch_test, num_workers=num_workers)

        test_dataset = Market1501(root + '/bounding_box_test', transform=test_transform, training=False, kpt_file=name+'-kpt.pkl' if args.pap else None)
        test_flip_dataset = Market1501(root + '/bounding_box_test', transform=test_flip_transform, training=False, kpt_file=name+'-kpt.pkl' if args.pap else None)
        test_loader = DataLoader(test_dataset, batch_size=batch_test, num_workers=num_workers)
        test_flip_loader = DataLoader(test_flip_dataset, batch_size=batch_test, num_workers=num_workers)
        return query_loader, query_flip_loader, test_loader, test_flip_loader

    def make_test_loader_MS_PR_PI(name):
        if name == 'msmt17':
            dclass = MSMT17
        elif name == 'partial_reid':
            dclass = PartialREID
        elif name == 'partial_ilids':
            dclass = PartialiLIDs
        else:
            raise ValueError('Invalid dataset name {}'.format(name))
        q_set = dclass(transform=test_transform, training=False, use_kpt=args.pap, use_ps=False, split='query')
        q_flip_set = dclass(transform=test_flip_transform, training=False, use_kpt=args.pap, use_ps=False, split='query')
        q_loader = DataLoader(q_set, batch_size=batch_test, num_workers=num_workers)
        q_flip_loader = DataLoader(q_flip_set, batch_size=batch_test, num_workers=num_workers)

        g_set = dclass(transform=test_transform, training=False, use_kpt=args.pap, use_ps=False, split='gallery')
        g_flip_set = dclass(transform=test_flip_transform, training=False, use_kpt=args.pap, use_ps=False, split='gallery')
        g_loader = DataLoader(g_set, batch_size=batch_test, num_workers=num_workers)
        g_flip_loader = DataLoader(g_flip_set, batch_size=batch_test, num_workers=num_workers)

        return q_loader, q_flip_loader, g_loader, g_flip_loader

    def make_test_loader(name):
        if name in ['market1501', 'cuhk03', 'duke']:
            return make_test_loader_M_C_D(get_dataset_root(name), name)
        elif name in ['msmt17', 'partial_reid', 'partial_ilids']:
            return make_test_loader_MS_PR_PI(name)

    test_loaders = [make_test_loader(name) for name in testset_names]

    mgn = MGN(len(train_dataset.unique_ids), args)
    if torch.cuda.device_count() > 1:
        mgn = nn.DataParallel(mgn)
    mgn = mgn.to(DEVICE)
    vanilla_cross_entropy_loss = nn.CrossEntropyLoss()
    cross_entropy_loss = nn.CrossEntropyLoss(reduce=False)
    triplet_semihard_loss = TripletSemihardLoss(margin=0.1, DEVICE = DEVICE, sampling = sampling, batch_id = batch_id, batch_image = batch_image)  #batch_hard, .'curriculum'
    ps_loss = PSLoss()

    optimizer_start1 = optim.SGD(mgn.parameters(), lr=lr_1, momentum=0.9, weight_decay=weight_decay)
    optimizer_start2 = optim.SGD(mgn.parameters(), lr=lr_2, momentum=0.9, weight_decay=weight_decay)
    scheduler_1 = optim.lr_scheduler.MultiStepLR(optimizer_start1, [140, 180], gamma=0.1)
    scheduler_2 = optim.lr_scheduler.MultiStepLR(optimizer_start2, [140, 180], gamma=0.1)   # best [140, 180] [120, 160]

    def get_model_input(inputs, target):
        dic = {'im': inputs.to(DEVICE)}
        if 'pap_mask_2p' in target:
            dic['pap_mask_2p'] = target['pap_mask_2p'].to(DEVICE)
            dic['pap_mask_3p'] = target['pap_mask_3p'].to(DEVICE)
        return dic

    def extract_loader_feat(loader, verbose=False):
        feat = []
        vis = []
        i = 0
        for inputs, target in loader:
            if verbose:
                print(i)
                i += 1
            with torch.no_grad():
                output = mgn(get_model_input(inputs, target))
            feat.append(output[1].detach().cpu().numpy())
            if args.pap:
                vis_ = np.concatenate([np.ones([len(output[1]), 3]), torch.stack(output[5+3+5:5+3+5+5], 1).detach().cpu().numpy()], 1)
                vis.append(vis_)
        feat = np.concatenate(feat)
        vis = np.concatenate(vis) if args.pap else None
        return feat, vis

    def test(query_loader, query_flip_loader, test_loader, test_flip_loader, trainset_name, testset_name, epoch, verbose=False):
        cache_file = '{}/feat_cache-{}_to_{}.pkl'.format(exp_dir, trainset_name, testset_name)
        if args.use_feat_cache:
            assert os.path.exists(cache_file), "Feature cache file {} does not exist!".format(cache_file)
            query_2, q_vis, query_flip_2, q_vis, test_2, test_vis, test_flip_2, test_vis, q_ids, q_cams, g_ids, g_cams = load_pickle(cache_file)
        else:
            query_2, q_vis = extract_loader_feat(query_loader, verbose=verbose)
            query_flip_2, q_vis = extract_loader_feat(query_flip_loader, verbose=verbose)

            test_2, test_vis = extract_loader_feat(test_loader, verbose=verbose)
            test_flip_2, test_vis = extract_loader_feat(test_flip_loader, verbose=verbose)

            q_ids = query_loader.dataset.ids
            q_cams = query_loader.dataset.cameras
            g_ids = test_loader.dataset.ids
            g_cams = test_loader.dataset.cameras
            save_pickle([query_2, q_vis, query_flip_2, q_vis, test_2, test_vis, test_flip_2, test_vis, q_ids, q_cams, g_ids, g_cams], cache_file)

        if args.test_which_feat > 0:
            # TODO: implement for pap
            idx = args.test_which_feat
            query_2 = query_2[:, 256 * idx - 256:256 * idx]
            query_flip_2 = query_flip_2[:, 256 * idx - 256:256 * idx]
            test_2 = test_2[:, 256 * idx - 256:256 * idx]
            test_flip_2 = test_flip_2[:, 256 * idx - 256:256 * idx]

        query = normalize(query_2 + query_flip_2)
        test = normalize(test_2 + test_flip_2)

        if verbose:
            print('query.shape:', query.shape)
            print('test.shape:', test.shape)
            if args.pap:
                print('q_vis.shape:', q_vis.shape)
                print('test_vis.shape:', test_vis.shape)

        if args.pap:
            dist_1 = compute_dist_with_visibility(query, test, q_vis, test_vis, dist_type='euclidean', avg_by_vis_num=False)
        else:
            dist_1 = cdist(query, test)
        r_1 = cmc(dist_1, q_ids, g_ids, q_cams, g_cams,
                  separate_camera_set=False,
                  single_gallery_shot=False,
                  first_match_break=True)
        m_ap_1 = mean_ap(dist_1, q_ids, g_ids, q_cams, g_cams)
        print('EPOCH [%d] %s -> %s: mAP=%f, r@1=%f, r@3=%f, r@5=%f, r@10=%f' % (epoch + 1, trainset_name, testset_name, m_ap_1, r_1[0], r_1[2], r_1[4], r_1[9]))

    if args.only_test:
        mgn.eval()
        if not args.use_feat_cache:
            if args.model_weight_file:
                model_weight_file = args.model_weight_file
            else:
                model_weight_file = '{}/model_weight.pth'.format(exp_dir)
            load_model_weight((mgn.module if hasattr(mgn, 'module') else mgn), model_weight_file)
        for name, test_loader in zip(testset_names, test_loaders):
            test(test_loader[0], test_loader[1], test_loader[2], test_loader[3], trainset_name, name, -1, verbose=False)
        exit()

    for epoch in range(epochs):
        mgn.train()
        scheduler_1.step()
        scheduler_2.step()
        running_loss = 0.0
        running_loss_1 = 0.0
        running_loss_2 = 0.0
        if epoch < 20:
            optimizer_1 = optim.SGD(mgn.parameters(), lr=0.01+0.0045*epoch, momentum=0.9, weight_decay=weight_decay)
            optimizer_2 = optim.SGD(mgn.parameters(), lr=0.001+0.00045*epoch, momentum=0.9, weight_decay=weight_decay) 
        else:
            optimizer_1 = optimizer_start1
            optimizer_2 = optimizer_start2
            
        for i, data in enumerate(train_loader):
            inputs, target = data
            inputs = inputs.to(DEVICE)
            for k, v in target.items():
                target[k] = v.to(DEVICE)
            labels = target['id']
            outputs = mgn(get_model_input(inputs, target))
            optimizer_1.zero_grad()
            if args.pap:
                losses_1 = [vanilla_cross_entropy_loss(output, labels) for output in outputs[5:5+3]] + [(cross_entropy_loss(output, labels) * v).sum() / (v.sum() + 1e-12) for output, v in zip(outputs[5+3:5+3+5], outputs[5+3+5:5+3+5+5])]
            else:
                losses_1 = [vanilla_cross_entropy_loss(output, labels) for output in outputs[5:5+8]]
            loss_1 = sum(losses_1) / len(losses_1)
            psl = 0
            if args.src_ps_lw > 0:
                psl = (ps_loss(outputs[-3], target['ps_label']) + ps_loss(outputs[-2], target['ps_label']) + ps_loss(outputs[-1], target['ps_label'])) / 3.
            (loss_1 + psl * args.src_ps_lw).backward()
            if args.cd_ps_lw > 0:
                cd_inputs, cd_targets = cd_train_loader.next_batch()
                cd_inputs = cd_inputs.to(DEVICE)
                for k, v in cd_targets.items():
                    cd_targets[k] = v.to(DEVICE)
                pap_old = args.pap
                args.pap = False
                outputs = mgn(get_model_input(cd_inputs, cd_targets))
                args.pap = pap_old
                cd_psl = (ps_loss(outputs[-3], cd_targets['ps_label']) + ps_loss(outputs[-2], cd_targets['ps_label']) + ps_loss(outputs[-1], cd_targets['ps_label'])) / 3.
                (cd_psl * args.cd_ps_lw).backward()
            optimizer_1.step()

            outputs = mgn(get_model_input(inputs, target))
            optimizer_2.zero_grad()
            losses_2 = [triplet_semihard_loss(output, labels, epoch) for output in outputs[2:5]]
            loss_2 = sum(losses_2) / len(losses_2)
            psl = 0
            if args.src_ps_lw > 0:
                psl = (ps_loss(outputs[-3], target['ps_label']) + ps_loss(outputs[-2], target['ps_label']) + ps_loss(outputs[-1], target['ps_label'])) / 3.
            (loss_2 + psl * args.src_ps_lw).backward()
            if args.cd_ps_lw > 0:
                cd_inputs, cd_targets = cd_train_loader.next_batch()
                cd_inputs = cd_inputs.to(DEVICE)
                for k, v in cd_targets.items():
                    cd_targets[k] = v.to(DEVICE)
                pap_old = args.pap
                args.pap = False
                outputs = mgn(get_model_input(cd_inputs, cd_targets))
                args.pap = pap_old
                cd_psl = (ps_loss(outputs[-3], cd_targets['ps_label']) + ps_loss(outputs[-2], cd_targets['ps_label']) + ps_loss(outputs[-1], cd_targets['ps_label'])) / 3.
                (cd_psl * args.cd_ps_lw).backward()
            optimizer_2.step()

            running_loss_1 += loss_1.item()
            running_loss_2 += loss_2.item()
            running_loss = running_loss + (loss_1.item() + loss_2.item())/2.0

            print('%d/%d - %d/%d - loss: %f - ps_loss: %f - cd_ps_loss: %f' % (epoch + 1, epochs, i, len(train_loader), (loss_1.item() + loss_2.item())/2, psl.item() if isinstance(psl, torch.Tensor) else 0, cd_psl.item() if args.cd_ps_lw > 0 else 0))
        print('epoch: %d/%d - loss1:      %f' % (epoch + 1, epochs, running_loss_1 / len(train_loader)))
        print('epoch: %d/%d - loss2:      %f' % (epoch + 1, epochs, running_loss_2 / len(train_loader)))

        # if (epoch + 1) % 50 == 0:
        #     model_weight_file = '{}/model_weight.pth'.format(exp_dir)
        #     save_model(mgn, model_weight_file)
        #     mgn.eval()
        #     for name, test_loader in zip(testset_names, test_loaders):
        #         test(test_loader[0], test_loader[1], test_loader[2], test_loader[3], trainset_name, name, epoch)
    model_weight_file = '{}/model_weight.pth'.format(exp_dir)
    save_model(mgn, model_weight_file)
    mgn.eval()
    for name, test_loader in zip(testset_names, test_loaders):
        test(test_loader[0], test_loader[1], test_loader[2], test_loader[3], trainset_name, name, epoch)


class CommaSeparatedSeq(object):
    def __init__(self, seq_class=tuple, func=int):
        self.seq_class = seq_class
        self.func = func

    def __call__(self, s):
        return self.seq_class([self.func(i) for i in s.split(',')])


def str2bool(v):
    """From https://github.com/amdegroot/ssd.pytorch"""
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    print('Used Python:', sys.executable)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--gpuId', type=str, default='0', help='input gpu id')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='input training epochs')
    parser.add_argument('-w', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_id', type=int, default=2)
    parser.add_argument('--batch_image', type=int, default=4)
    parser.add_argument('--lr_1', type=float, default = .1)
    parser.add_argument('--lr_2', type=float, default = .01)
    parser.add_argument('--rand_crop', type=eval, default=True, help='Either True or False')
    parser.add_argument('--erasing_p', type=float, default = 0.5)
    parser.add_argument('--sampling', type=str, default = 'batch_hard')
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--trainset_name', type=str)
    parser.add_argument('--cd_trainset_name', type=str)
    parser.add_argument('--cd_train_batch_size', type=int, default=16*8)
    parser.add_argument('--head_1part_stride', type=int, default=2)
    parser.add_argument('--pap', type=eval, default=False, help='Either True or False')
    parser.add_argument('--src_ps_lw', type=float, default=0)
    parser.add_argument('--cd_ps_lw', type=float, default=0)
    parser.add_argument('--only_test', type=eval, default=False, help='Either True or False')
    parser.add_argument('--model_weight_file', type=str, default='')
    parser.add_argument('--testset_names', type=CommaSeparatedSeq(list, str), default=['market1501', 'cuhk03', 'duke', 'msmt17'])
    parser.add_argument('--use_feat_cache', type=str2bool, default=False)
    parser.add_argument('--test_which_feat', type=int, default=-1, help='Either -1 or one of 1,2,3,4,5,6,7,8')

    args = parser.parse_args()
    print(args)
    time_start = time.time()
    run(args)
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
