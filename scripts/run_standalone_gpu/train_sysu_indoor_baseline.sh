#!/usr/bin/env bash
myfile="train_sysu_indoor_baseline.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python train.py \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--gpu 0 \
--device-target GPU \
--pretrain "resnet50.ckpt" \
--tag "sysu_indoor_baseline" \
--data-path "Define your own path/sysu" \
--loss-func "id+tri" \
--branch main \
--sysu-mode "indoor" \
--part 0 \
--epoch 40