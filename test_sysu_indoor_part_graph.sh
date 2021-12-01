export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

python test_ddag.py \
--dataset SYSU \
--gpu 2 \
--device-target GPU \
--resume "/home/shz/pytorch/zzw/DDAG_mindspore_modify/logs/sysu_all_part_graph/training/epoch_35_rank1_56.68_mAP_55.72_test_SYSU_batch-size_2*8*4=64_adam_lr_0.0035_loss-func_id+tri_P_3_Graph__main.ckpt" \
--tag "sysu_indoor_part_graph" \
--data-path "/home/shz/pytorch/data/sysu" \
--branch main \
--sysu-mode "" \
--part 3