#!/bin/bash

myfile="train_distributed.sh"

if [ ! -f "$myfile"]; then
    echo "Please first enter DDAG_mindspore/scripts/run_distributed_train and run. Exit..."
    exit 0
fi

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash train_dirtributed.sh RANK_SIZE"
echo "For example: bash run.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

RANK_SIZE=$1

EXEC_PATH=$(pwd)

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

cd ../..

for((i=1; i < ${RANK_SIZE}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    python train.py \
        --MSmode "GRAPH_MODE" \
        --dataset SYSU \
        --optim adam \
        --lr 0.0035 \
        --device-target Ascend \
        --device-id ${DEVICE_ID} \
        --run-distribute \
        --pretrain "/opt_data/ecnu/resnet50.ckpt" \
        --tag "sysu_all_part_graph" \
        --data-path "/opt_data/ecnu/dataset/sysu" \
        --loss-func "id+tri" \
        --branch main \
        --sysu-mode "all" \
        --part 3 \
        --graph \
        --epoch 40 \
    2>&1 >> logs/sysu_all_part_graph/distributed_training/device${i}.log
done

python train.py \
--MSmode "GRAPH_MODE" \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--device-target Ascend \
--device-id 0 \
--run-distribute \
--pretrain "/opt_data/ecnu/resnet50.ckpt" \
--tag "sysu_all_part_graph" \
--data-path "/opt_data/ecnu/dataset/sysu" \
--loss-func "id+tri" \
--branch main \
--sysu-mode "all" \
--part 3 \
--graph \
--epoch 40
2>&1 >> logs/sysu_all_part_graph/distributed_training/device${i}.log

export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"



if [ $? -eq 0 ];then
    echo "distributed training success"
else
    echo "distributed training failed"
    exit 2
fi

