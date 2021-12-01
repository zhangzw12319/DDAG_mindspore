export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

python train_ddag.py \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--gpu 3 \
--device-target GPU \
--pretrain "model/pretrain/resnet50_ascend_v111_imagenet2012_official_cv_bs32_acc76/resnet50.ckpt" \
--tag "sysu_indoor_baseline" \
--data-path "/home/shz/pytorch/data/sysu" \
--loss-func "id+tri" \
--branch main \
--sysu-mode "indoor" \
--part 0