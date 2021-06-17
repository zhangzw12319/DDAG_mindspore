export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

python train_ddag.py \
--dataset SYSU \
--data_path SYSU-MM01 \
--lr 0.1 \
--graph \
--part 0 \
--gpu 0 \
--device_target GPU