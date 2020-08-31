import numpy as np
from PIL import Image

############
# Fuse Parts

# Original 7 parts
#   'background': 0,
#   'head': 1,
#   'torso': 2,
#   'upper_arm': 3,
#   'lower_arm': 4,
#   'upper_leg': 5,
#   'lower_leg': 6,
#   'foot': 7,
fuse_mapping = {
    '4parts': [(0, 0), (1, 1), (2, 2), (3, 2), (4, 2), (5, 3), (6, 4), (7, 4)],
    '2parts': [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 2), (6, 2), (7, 2)],
    'fg': [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
}

def fuse_parts(ps_label, fuse_type):
    ori_ps_label = np.array(ps_label)
    ps_label = ori_ps_label.copy()
    for m in fuse_mapping[fuse_type]:
        ps_label[ori_ps_label == m[0]] = m[1]
    ps_label = Image.fromarray(ps_label, mode='L')
    return ps_label