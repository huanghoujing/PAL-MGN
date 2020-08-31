from __future__ import print_function
import os.path as osp
import numpy as np
import torch
from file_utils import load_pickle, save_pickle


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def array_str(array, fmt='{:.2f}', sep=', ', with_boundary=True):
    """String of a 1-D tuple, list, or numpy array containing digits."""
    ret = sep.join([fmt.format(float(x)) for x in array])
    if with_boundary:
        ret = '[' + ret + ']'
    return ret


def array_2d_str(array, fmt='{:.2f}', sep=', ', row_sep='\n', with_boundary=True):
    """String of a 2-D tuple, list, or numpy array containing digits."""
    ret = row_sep.join([array_str(x, fmt=fmt, sep=sep, with_boundary=with_boundary) for x in array])
    if with_boundary:
        ret = '[' + ret + ']'
    return ret


def _cal_cross_part_sim(feat_list):
    """
    Args:
        feat_list: a list of numpy array, each with shape [N, d]
    Returns:
        sim_mat: a numpy array with shape [n_parts, n_parts], sim_mat[i, j]
            is similarity between feat_list[i] and feat_list[j], averaged across samples
    """
    # [N, n_parts, d]
    feat = np.stack(feat_list, 1)
    feat = normalize(feat, axis=2)
    # print('feat.shape:', feat.shape)
    # [N, n_parts, n_parts]
    sim_mat = np.matmul(feat, feat.transpose(0, 2, 1))
    sim_mat = sim_mat.mean(axis=0)
    return sim_mat


def cal_cross_part_sim(feat_cache_file):
    query_2, q_vis, query_flip_2, q_vis, test_2, test_vis, test_flip_2, test_vis, q_ids, q_cams, g_ids, g_cams = load_pickle(feat_cache_file)
    feat_list = [query_2[:, -256*3:-256*2], query_2[:, -256*2:-256], query_2[:, -256:]]
    sim_mat = _cal_cross_part_sim(feat_list)
    return sim_mat


if __name__ == '__main__':
    try:
        sim_mat = cal_cross_part_sim('exp/train_mgn/msmt17_run2/feat_cache-msmt17_to_msmt17.pkl')
        print('MGN Part Similarity:\n{}'.format(array_2d_str(sim_mat, fmt='{:.4f}', with_boundary=True, sep=', ', row_sep=',\n')))
    except:
        pass

    try:
        sim_mat = cal_cross_part_sim('exp/train_mgn_ps/ps_lw_1-PartSegHeadDeconvConv-ps_fuse_type_None/msmt17_run2/feat_cache-msmt17_to_msmt17.pkl')
        print('MGN-S-PS Part Similarity:\n{}'.format(array_2d_str(sim_mat, fmt='{:.4f}', with_boundary=True, sep=', ', row_sep=',\n')))
    except:
        pass