#!/usr/bin/env bash
myfile="train_sysu_indoor_part_graph.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python train.py \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--device-id 1 \
--device-target Ascend \
--pretrain "resnet50.ckpt" \
--tag "sysu_indoor_part_graph" \
--data-path "Define your own path/sysu" \
--loss-func "id+tri" \
--branch main \
--sysu-mode "indoor" \
--part 3 \
--graph True \
--epoch 40