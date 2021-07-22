export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

python train_ddag.py \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--gpu 2 \
--device-target GPU \
--data-path "/home/shz/pytorch/data/" \
--loss-func "id" \
--batch-size 16

# --resume "./save_checkpoints/mAP_0.2056_rank1_0.2103_Exp_0_SYSU_batch-size_8*8=64_lr_0.0035_loss-func_id_adam_master.ckpt" \
