export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

python train_ddag.py \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--gpu 1 \
--device_target GPU \
--ckpt id
