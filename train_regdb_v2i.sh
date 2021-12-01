export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

python train_ddag.py \
--dataset RegDB \
--optim adam \
--lr 0.0035 \
--gpu 1 \
--device-target GPU \
--pretrain "model/pretrain/resnet50_ascend_v111_imagenet2012_official_cv_bs32_acc76/resnet50.ckpt" \
--tag "regdb_v2i" \
--data-path "/home/shz/pytorch/data/regdb" \
--loss-func "id+tri" \
--branch main \
--regdb-mode "v2i" \
--part 0 \
--graph