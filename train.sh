export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

python train_ddag_debug.py \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--gpu 3 \
--device_target GPU \
--data_path "/home/shz/pytorch/data/"
--ckpt "id"
