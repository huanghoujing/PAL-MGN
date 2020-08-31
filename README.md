#### About

This is the official implementation of `MGN vs. MGN+PS` for paper **Improve Person Re-Identification With Part Awareness Learning**, TIP 2020.

```
@article{huang2020improve,
  title={Improve Person Re-Identification With Part Awareness Learning},
  author={Huang, Houjing and Yang, Wenjie and Lin, Jinbin and Huang, Guan and Xu, Jiamiao and Wang, Guoli and Chen, Xiaotang and Huang, Kaiqi},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={7468--7481},
  year={2020},
  publisher={IEEE}
}
```


#### Requirements
- Python 3
- Pytorch 0.4.1
- Torchvision 0.2.1
- No special requirement for sklearn version
- 4 GPUs

#### Datasets

- Market1501
- CUHK03
- Duke
- MSMT17

#### Dataset Path

Under project dir
- Market-1501-v15.09.15
- Market-1501-v15.09.15_ps_label
- cuhk03-np-jpg/detected
  - bounding_box_train
  - query
  - bounding_box_test
- cuhk03-np-jpg_ps_label/detected
  - bounding_box_train
  - query
  - bounding_box_test
- DukeMTMC-reID
- DukeMTMC-reID_ps_label
- msmt17/MSMT17_V1
- msmt17/MSMT17_V1_ps_label

The part segmentation labels (only used during training) for datasets can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1Mm2gWO-Xg3wiyCd6SEAWaA) or [Google Drive](https://drive.google.com/open?id=1BARSoobjTAPeOSOM-HnGzlOYTj1l9-Qs).

The JPEG images of CUHK03-NP (`cuhk03-np-jpg/detected`) can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1ha9uAtVzX1hFG3piqcdvCg) (password `vtjp`) or [Google Drive](https://drive.google.com/drive/folders/1lGaQ3I9eYtBEYHq2nYubuO_N0PeY4SKB?usp=sharing).

#### Models

- MGN: `bash train_mgn.sh`
- MGN+PS: `bash train_s_ps_erase_ps_label.sh`

#### Train Augmentation

- Flip
- Random Erasing

#### Test Augmentation

- Flip

#### Visualize Activation Map & Grad-cam

- visualize_v1.py

#### Examples

- Train `MGN` on Market1501
    ```bash
    gpus=0,1,2,3 python_exc=python train_set=market1501 testset_names=market1501 run=_run1 bash train_mgn.sh;
    ```

- Train `MGN+PS` on Market1501
    ```bash
    gpus=0,1,2,3 python_exc=python train_set=market1501 testset_names=market1501 run=_run1 bash train_s_ps_erase_ps_label.sh;
    ```

- Train `MGN` on MSMT17, test on Market1501, CUHK03 and MSMT17
    ```bash
    gpus=0,1,2,3 python_exc=python train_set=msmt17 testset_names=market1501,cuhk03,msmt17 run=_run1 bash train_mgn.sh;
    ```

- Test `MGN` MSMT17->MSMT17. Make sure you have downloaded the model weight, [Baidu Cloud](https://pan.baidu.com/s/1GRUe8w9YPDJL4q3vV_oj1w) (password `l5vk`) or [Google Drive](https://drive.google.com/drive/folders/1I7U9BNdRJbavsvGxJsPz0z0Qf4kT9Wgz?usp=sharing), placing it to `exp/train_mgn/msmt17/model_weight.pth`.
    ```bash
    gpus=0,1,2,3 python_exc=python train_set=msmt17 testset_names=msmt17 only_test=True bash train_mgn.sh;
    ```
    You should get score `mAP=0.560998, r@1=0.801269, r@3=0.865597, r@5=0.888155, r@10=0.916459`.

- Test `MGN+PS` MSMT17->MSMT17. Make sure you have downloaded the model weight, [Baidu Cloud](https://pan.baidu.com/s/1EFIHrSNj1m84LYkWJsD-cw) (password `3u8a`) or [Google Drive](https://drive.google.com/drive/folders/1z5xs1WIg56CwMiE3Dtx-Mef6PG7qVLFP?usp=sharing), placing it to `exp/train_mgn_ps/ps_lw_1-PartSegHeadDeconvConv-ps_fuse_type_None/msmt17/model_weight.pth`.
    ```bash
    gpus=0,1,2,3 python_exc=python train_set=msmt17 testset_names=msmt17 only_test=True bash train_s_ps_erase_ps_label.sh;
    ```
    You should get score `mAP=0.623187, r@1=0.841324, r@3=0.896475, r@5=0.914229, r@10=0.934385`.

- Test `MGN` MSMT17->MSMT17, using only the first part feature (Paper Figure 13(b)).
    ```bash
    gpus=0,1,2,3 python_exc=python train_set=msmt17 testset_names=msmt17 only_test=True use_feat_cache=True test_which_feat=1 bash train_mgn.sh;
    ```

- Test `MGN+PS` MSMT17->MSMT17, using only the first part feature (Paper Figure 13(b)).
    ```bash
    gpus=0,1,2,3 python_exc=python train_set=msmt17 testset_names=msmt17 only_test=True use_feat_cache=True test_which_feat=1 bash train_s_ps_erase_ps_label.sh;
    ```

#### Code Clarity

Note that
- The content about `keypoints`, `Part Aligned Pooling (PAP)`, `occluded ReID datasets`, `cd_ps_lw` are not related to the paper, and can simply be ignored, when you are reading the code.
- This repository is mainly for reproducibility, not for efficient engineering exploitation.
- To avoid introducing new bugs, I did not clean up this part of code.

#### Acknowledgement

The implementation of MGN is originated from [MGN-pytorch](https://github.com/seathiefwang/MGN-pytorch), with some modifications.