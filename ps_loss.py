from __future__ import print_function
import torch


class PSLoss(object):
    def __init__(self, num_classes=8):
        self.criterion = torch.nn.CrossEntropyLoss(reduce=False)
        self.num_classes = num_classes

    # TODO: Pytorch newer versions support high-dimension CrossEntropyLoss, so no need to reshape pred and label.
    def __call__(self, ps_pred, ps_label):

        # Calculation
        N, C, H, W = ps_pred.size()
        assert ps_label.size() == (N, H, W)
        # shape [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
        ps_pred = ps_pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # shape [N, H, W] -> [NHW]
        ps_label = ps_label.view(N * H * W).detach()
        loss = self.criterion(ps_pred, ps_label)
        # Calculate each class avg loss and then average across classes, to compensate for classes that have few pixels
        loss_ = 0
        cur_batch_n_classes = 0
        for i in range(self.num_classes):
            loss_i = loss[ps_label == i]
            if loss_i.numel() > 0:
                loss_ += loss_i.mean()
                cur_batch_n_classes += 1
        loss_ /= (cur_batch_n_classes + 1e-8)
        loss = loss_
        return loss
