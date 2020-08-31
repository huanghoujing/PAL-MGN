export learning_rate_1=0.1
export learning_rate_2=0.01
export gpu_id=0
export epoch_number=200
export weight_decay=5e-4
export batch_id=16
export batch_image=8
export rand_crop=False # hhj
export random_erase=0.5 # hhj
export sampling='batch_hard' #curriculum, batch_hard
export head_1part_stride=1 # hhj
export train_set=${train_set:=msmt17} # market1501, cuhk03, msmt17
export testset_names=${testset_names:=msmt17} # market1501, cuhk03, msmt17
export use_feat_cache=${use_feat_cache:=False} # False, True
export only_test=${only_test:=False} #hhj
export test_which_feat=${test_which_feat:=-1} #hhj
export python_exc=${python_exc:=python3} #hhj
export gpus=${gpus:=0,1,2,3} #hhj

export model_weight_file=""

# ${run} can be empty, or set to e.g. "_run1", "_run2", etc
exp_dir=exp/train_mgn/${train_set}${run}

mkdir -p ${exp_dir}

if [ "${only_test}" == True ]; then
    log_file="${exp_dir}/test_log-$(date +%Y%m%d%H%M%S).txt"
else
    log_file="${exp_dir}/train_log-$(date +%Y%m%d%H%M%S).txt"
fi

echo "Used GPUs: ${gpus}"

CUDA_VISIBLE_DEVICES=${gpus} ${python_exc} mgn_pap_ps.py \
--gpuId ${gpu_id} \
--epochs ${epoch_number} \
--weight_decay ${weight_decay} \
--batch_id ${batch_id} \
--batch_image ${batch_image} \
--lr_1 ${learning_rate_1} \
--lr_2 ${learning_rate_2} \
--rand_crop ${rand_crop} \
--erasing_p ${random_erase} \
--sampling ${sampling} \
--exp_dir ${exp_dir} \
--trainset_name ${train_set} \
--testset_names ${testset_names} \
--head_1part_stride ${head_1part_stride} \
--only_test ${only_test} \
--test_which_feat ${test_which_feat} \
--use_feat_cache ${use_feat_cache} \
--model_weight_file "${model_weight_file}" \
2>&1 | tee ${log_file}