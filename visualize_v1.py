from __future__ import print_function

import sys
import argparse
import copy
import os
import os.path as osp
import time
import datetime

import cv2
import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet50, Bottleneck
from torchvision.transforms import functional
import torch.nn.functional as F

from msmt17 import MSMT17
from file_utils import load_pickle, save_pickle
from utils import load_state_dict
from utils import set_random_seed
from image_utils import heatmap_to_color_im, make_im_grid, restore_im, save_im


class MGN(nn.Module):
    def __init__(self, num_classes, args):
        super(MGN, self).__init__()

        self.args = args
        resnet = resnet50(pretrained=False)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0]
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        print('Model Structure:')
        print(self)

    def forward(self, in_dict):
        x = self.backbone(in_dict['im'])
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        return p1, p2, p3


class MGNGradCAM(nn.Module):
    def __init__(self, num_classes, args):
        super(MGNGradCAM, self).__init__()

        self.args = args
        resnet = resnet50(pretrained=False)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0]
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))

        self.fc_id_2048_0_tmp = nn.Linear(2048, 2048)
        self.fc_id_2048_1_tmp = nn.Linear(2048, 2048)
        self.fc_id_2048_2_tmp = nn.Linear(2048, 2048)
        self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_1 = nn.Linear(2048, num_classes)
        self.fc_id_2048_2 = nn.Linear(2048, num_classes)

        print('Model Structure:')
        print(self)

    def forward(self, in_dict):
        x = self.backbone(in_dict['im'])
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)
        l_p1 = self.fc_id_2048_0_tmp(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1_tmp(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2_tmp(zg_p3.squeeze(dim=3).squeeze(dim=2))
        l_p1 = self.fc_id_2048_0(l_p1)
        l_p2 = self.fc_id_2048_1(l_p2)
        l_p3 = self.fc_id_2048_2(l_p3)
        return l_p1, l_p2, l_p3


def load_model_weight(model, model_weight_file):
    assert osp.exists(model_weight_file), "model_weight_file {} does not exist!".format(model_weight_file)
    assert osp.isfile(model_weight_file), "model_weight_file {} is not file!".format(model_weight_file)
    model_weight = torch.load(model_weight_file, map_location=(lambda storage, loc: storage))
    load_state_dict(model, model_weight)
    msg = '=> Loaded model_weight from {}'.format(model_weight_file)
    print(msg)


def save_avg_map(im_list, feat, save_path):
    im_list = [restore_im(im, [0.229, 0.224, 0.225], [0.486, 0.459, 0.408], transpose=False, resize_w_h=(64, 128)) for im in im_list]
    # min_max_val = np.stack(feat).min(), np.stack(feat).max()
    min_max_val = None
    avg_feat = [f.mean(axis=0) for f in feat]
    avg_hmaps = [heatmap_to_color_im(f, normalize=True, min_max_val=min_max_val, resize=True, resize_w_h=(64, 128), transpose=True) for f in avg_feat]
    # ims = [read_im(p, resize_h_w=(128, 64), transpose=True) for p in im_paths]
    vis_im_list = []
    for im, ah in zip(im_list, avg_hmaps):
        vis_im_list.extend([im, ah])
    n_cols = 16
    n_rows = int(np.ceil(1. * len(vis_im_list) / n_cols))
    im = make_im_grid(vis_im_list, n_rows, n_cols, 8, 255)
    print('im.shape:', im.shape)
    save_im(im, save_path, transpose=True)


def get_feat_grad(model, batch, modules):
    """For an image."""
    model.eval()
    handles = []
    feat = []
    for m in modules:
        def func(m, i, o):
            # print("type(o)", type(o))
            feat.append(o)
        handles.append(m.register_forward_hook(func))
    grad = []
    for m in modules:
        def func(m, gi, go):
            # print("type(gi) {}, type(go) {}".format(type(gi), type(go)))
            # print("len(gi) {}, len(go) {}".format(len(gi), len(go)))
            # print("gi[0].shape {}, go[0].shape {}".format(gi[0].shape, go[0].shape))
            grad.append(go[0])
        handles.append(m.register_backward_hook(func))
    model.zero_grad()
    model(batch)[1][0][batch['label'][0]].backward()
    for h in handles:
        h.remove()
    return feat, grad


# The same as `cam = (cam - cam_min) / (cam_max - cam_min)`
# def get_grad_cam(feat, grad):
#     # cam = F.relu((feat * grad.mean(2, keepdim=True).mean(3, keepdim=True).expand_as(feat)).sum(1))  # N,H,W
#     # cam = (feat * grad.abs().mean(2, keepdim=True).mean(3, keepdim=True).expand_as(feat)).sum(1)  # N,H,W
#     # cam = grad.abs().sum(1)  # N,H,W
#     cam = F.relu(-grad).sum(1)  # N,H,W
#     # print("type(cam)", type(cam))
#     cam -= cam.min(1, keepdim=True)[0].min(2, keepdim=True)[0].expand_as(cam)
#     cam /= cam.max(1, keepdim=True)[0].max(2, keepdim=True)[0].expand_as(cam)
#     return cam


def get_grad_cam(feat, grad):
    # cam = F.relu((feat * grad.mean(2, keepdim=True).mean(3, keepdim=True).expand_as(feat)).sum(1))  # N,H,W
    cam = (feat * grad.abs().mean(2, keepdim=True).mean(3, keepdim=True).expand_as(feat)).sum(1)  # N,H,W
    # cam = grad.abs().sum(1)  # N,H,W
    # cam = F.relu(-grad).sum(1)  # N,H,W
    # cam = (F.relu(-grad) * (feat > 0).float()).sum(1)  # N,H,W
    # print("type(cam)", type(cam))
    cam_min = cam.min(1, keepdim=True)[0].min(2, keepdim=True)[0].expand_as(cam)
    cam_max = cam.max(1, keepdim=True)[0].max(2, keepdim=True)[0].expand_as(cam)
    cam = (cam - cam_min) / (cam_max - cam_min)
    return cam


# def show_cam_on_image(img, mask):
#     """
#     img: H x W x 3, GBR, 0~255 image
#     mask: H x W, 0~1 value
#     cam: H x W x 3, GBR, 0~255 image
#     """
#     heatmap = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
#     cam = np.float32(heatmap) + np.float32(img)
#     cam = 1. * cam / np.max(cam)
#     cam = np.uint8(255 * cam)
#     return cam

def show_cam_on_image(img, mask):
    """
    img: H x W x 3, RGB, 0~255 image
    mask: H x W, 0~1 value
    cam: H x W x 3, RGB, 0~255 image
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]  # RGB
    cam = np.float32(heatmap) + np.float32(img)
    cam = 1. * cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


@torch.no_grad()
def vis_act_map(args):
    set_random_seed(2)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dset = MSMT17(transform=transform, training=False, use_kpt=False, use_ps=False, split='query')
    loader = DataLoader(dset, batch_size=32, num_workers=4, shuffle=True)

    mgn = MGN(1041, args)
    # model_weight_file = 'exp/train_mgn/msmt17_run2/model_weight.pth'
    # save_path = 'exp/visualize/mgn_avg_map.png'

    model_weight_file = 'exp/train_mgn_ps/ps_lw_1-PartSegHeadDeconvConv-ps_fuse_type_None/msmt17_run2/model_weight.pth'
    save_path = 'exp/visualize/mgn_ps_avg_map.png'

    # save_path = os.path.join(args.exp_dir, 'mgn_avg_map.png')
    load_model_weight(mgn, model_weight_file)
    mgn = mgn.to(DEVICE)

    im_list, feat = [], []
    for inputs, target in loader:
        output = mgn({'im': inputs.to(DEVICE)})
        im_list.extend(inputs.cpu().numpy())
        feat.extend(output[1].cpu().numpy())
        break
    save_avg_map(im_list, feat, save_path)


def vis_grad_cam(args):
    set_random_seed(10)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dset = MSMT17(transform=transform, training=False, use_kpt=False, use_ps=False, split='train')
    loader = DataLoader(dset, batch_size=1, num_workers=4, shuffle=True)

    mgn = MGNGradCAM(1041, args)

    # model_weight_file = 'exp/train_mgn/msmt17_run2/model_weight.pth'
    # save_path = 'exp/visualize/grad_cam_mgn.png'

    # model_weight_file = 'exp/train_mgn_ps/ps_lw_1-PartSegHeadDeconvConv-ps_fuse_type_None/msmt17_run2/model_weight.pth'
    # save_path = 'exp/visualize/grad_cam_mgn_ps.png'

    # save_path = os.path.join(args.exp_dir, 'mgn_avg_map.png')
    load_model_weight(mgn, args.model_weight_file)
    mgn = mgn.to(DEVICE)

    vis_im_list = []
    for i, (inputs, target) in enumerate(loader):
        img = restore_im(inputs[0].cpu().numpy(), [0.229, 0.224, 0.225], [0.486, 0.459, 0.408], transpose=True, resize_w_h=(128, 256))
        vis_im_list += [img]
        batch = {'im': inputs.to(DEVICE), 'label': target['id'].to(DEVICE)}
        feat, grad = get_feat_grad(mgn, batch, [mgn.p2])  # much more obvious to see the difference between mgn and mgn_ps
        # feat, grad = get_feat_grad(mgn, batch, [mgn.p2[0]])  # less obvious
        cam = get_grad_cam(feat[0], grad[0])[0].detach().cpu().numpy()  # H,W
        cam = cv2.resize(cam, (128, 256), interpolation=cv2.INTER_LINEAR)
        cam = show_cam_on_image(img, cam)
        vis_im_list += [cam]
        if i+1 == 32:
            break
    vis_im_list = [im.transpose(2, 0, 1) for im in vis_im_list]
    n_cols = 16
    n_rows = int(np.ceil(1. * len(vis_im_list) / n_cols))
    im = make_im_grid(vis_im_list, n_rows, n_cols, 8, 255)
    save_im(im, args.save_path, transpose=True)


def vis_grad_cam_compare():
    set_random_seed(10)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dset = MSMT17(transform=transform, training=False, use_kpt=False, use_ps=False, split='train')
    loader = DataLoader(dset, batch_size=1, num_workers=4, shuffle=True)

    mgn = MGNGradCAM(1041, None)
    load_model_weight(mgn, 'exp/train_mgn/msmt17_run2/model_weight.pth')
    mgn = mgn.to(DEVICE)

    mgn_ps = MGNGradCAM(1041, None)
    load_model_weight(mgn_ps, 'exp/train_mgn_ps/ps_lw_1-PartSegHeadDeconvConv-ps_fuse_type_None/msmt17_run2/model_weight.pth')
    mgn_ps = mgn_ps.to(DEVICE)

    for i, (inputs, target) in enumerate(loader):
        vis_im_list = []
        img = restore_im(inputs[0].cpu().numpy(), [0.229, 0.224, 0.225], [0.486, 0.459, 0.408], transpose=True, resize_w_h=(128, 256))
        vis_im_list += [img]
        batch = {'im': inputs.to(DEVICE), 'label': target['id'].to(DEVICE)}

        feat, grad = get_feat_grad(mgn, batch, [mgn.p2])
        cam = get_grad_cam(feat[0], grad[0])[0].detach().cpu().numpy()  # H,W
        cam = cv2.resize(cam, (128, 256), interpolation=cv2.INTER_LINEAR)
        cam = show_cam_on_image(img, cam)
        vis_im_list += [cam]

        feat, grad = get_feat_grad(mgn_ps, batch, [mgn_ps.p2])
        cam = get_grad_cam(feat[0], grad[0])[0].detach().cpu().numpy()  # H,W
        cam = cv2.resize(cam, (128, 256), interpolation=cv2.INTER_LINEAR)
        cam = show_cam_on_image(img, cam)
        vis_im_list += [cam]

        vis_im_list = [im.transpose(2, 0, 1) for im in vis_im_list]
        im = make_im_grid(vis_im_list, 1, 3, 4, 255)
        save_im(im, 'exp/visualize/grad_cam_wo_w_ps/{:02d}.png'.format(i+1), transpose=True)

        if i+1 == 64:
            break



if __name__ == '__main__':
    print('Used Python:', sys.executable)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--model_weight_file', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')

    args = parser.parse_args()
    print(args)
    time_start = time.time()

    # vis_act_map(args)

    # args.model_weight_file = 'exp/train_mgn/msmt17_run2/model_weight.pth'
    # args.save_path = 'exp/visualize/grad_cam_mgn.png'
    # vis_grad_cam(args)
    #
    # args.model_weight_file = 'exp/train_mgn_ps/ps_lw_1-PartSegHeadDeconvConv-ps_fuse_type_None/msmt17_run2/model_weight.pth'
    # args.save_path = 'exp/visualize/grad_cam_mgn_ps.png'
    # vis_grad_cam(args)

    vis_grad_cam_compare()

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
