export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

python train_ddag.py \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--gpu 1 \
--device_target GPU \
--pretrained "/home/ubuntu/hdd1/shz/pytorch/shz/DDAG_mindspore_zzw/pretrained/resnet50_ascend_v111_imagenet2012_official_cv_bs32_acc76/resnet50.ckpt" \
--ckpt "ckpt_0"
