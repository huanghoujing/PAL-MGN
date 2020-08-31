from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch


class RandomErasingWithPS(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
         img: torch tensor with shape [3, h, w]
         ps: torch tensor with shape [h, w]
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, ps):

        assert len(img.shape) == 3
        assert img.shape[0] == 3
        im_h, im_w = img.shape[1:]
        ps_h, ps_w = ps.shape
        if random.uniform(0, 1) > self.probability:
            return img, ps

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1. / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(1. * target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                ps_x1 = min(max(0, int(1. * x1 * ps_h / im_h)), ps_h)
                ps_x2 = min(max(0, int(1. * (x1 + h) * ps_h / im_h)), ps_h)
                ps_y1 = min(max(0, int(1. * y1 * ps_w / im_w)), ps_w)
                ps_y2 = min(max(0, int(1. * (y1 + w) * ps_w / im_w)), ps_w)
                ps[ps_x1:ps_x2, ps_y1:ps_y2] = 0
                return img, ps

        return img, ps
