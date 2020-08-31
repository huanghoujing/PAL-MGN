from __future__ import print_function
from __future__ import division
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import average_precision_score

import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

from torchvision.transforms import functional as F


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]):
            continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap_deprecated(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    # -------------------------------------------------------------------------
    # The behavior of method `sklearn.average_precision` has changed since version
    # 0.19.
    # Version 0.18.1 has same results as Matlab evaluation code by Zhun Zhong
    # (https://github.com/zhunzhong07/person-re-ranking/
    # blob/master/evaluation/utils/evaluation.m) and by Liang Zheng
    # (http://www.liangzheng.org/Project/project_reid.html).
    # My current awkward solution is sticking to this older version.
    import sklearn
    cur_version = sklearn.__version__
    required_version = '0.18.1'
    if cur_version != required_version:
        print('User Warning: Version {} is required for package scikit-learn, '
              'your current version is {}. '
              'As a result, the mAP score may not be totally correct. '
              'You can try `pip uninstall scikit-learn` '
              'and then `pip install scikit-learn=={}`'.format(
            required_version, cur_version, required_version))
    # -------------------------------------------------------------------------
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


# hhj
# Modified from https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/evaluate.py
def ap_zzd(y_true, y_score):
    ngood = y_true.sum()
    d_recall = 1.0 / ngood
    rows_good = np.argwhere(y_true).flatten()
    ap = 0
    for i in range(ngood):
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap


def _to_array(x):
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    assert isinstance(x, np.ndarray), "Type of input is {}".format(type(x))
    return x


def mean_ap(
        distmat,
        query_ids=None,
        gallery_ids=None,
        query_cams=None,
        gallery_cams=None,
        average=True):
    """
    Args:
        distmat: numpy array with shape [num_query, num_gallery], the
            pairwise distance between query and gallery samples
        query_ids: numpy array with shape [num_query]
        gallery_ids: numpy array with shape [num_gallery]
        query_cams: numpy array with shape [num_query]
        gallery_cams: numpy array with shape [num_gallery]
        average: whether to average the results across queries
    Returns:
        If `average` is `False`:
            ret: numpy array with shape [num_query]
            is_valid_query: numpy array with shape [num_query], containing 0's and
                1's, whether each query is valid or not
        If `average` is `True`:
            a scalar
    """

    # Ensure numpy array
    assert isinstance(distmat, np.ndarray)
    query_ids = _to_array(query_ids)
    gallery_ids = _to_array(gallery_ids)
    query_cams = _to_array(query_cams)
    gallery_cams = _to_array(gallery_cams)

    m, n = distmat.shape

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = np.zeros(m)
    is_valid_query = np.zeros(m)
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        is_valid_query[i] = 1
        aps[i] = ap_zzd(y_true, y_score)
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    if average:
        return float(np.sum(aps)) / np.sum(is_valid_query)
    return aps, is_valid_query