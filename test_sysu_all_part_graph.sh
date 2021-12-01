export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

python test_ddag.py \
--dataset SYSU \
--gpu 2 \
--device-target GPU \
--resume "logs/sysu_all_part_graph/training/epoch_25_rank1_59.84_mAP_57.31_SYSU_batch-size_2*8*4=64_adam_lr_0.0035_loss-func_id+tri_P_3_Graph__main.ckpt" \
--tag "sysu_all_part_graph" \
--data-path "/home/shz/pytorch/data/sysu" \
--branch main \
--sysu-mode "all" \
--part 3