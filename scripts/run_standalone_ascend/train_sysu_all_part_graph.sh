myfile="train_sysu_all_part_graph.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python train.py \
--MSmode "GRAPH_MODE" \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--device-id 1 \
--device-target Ascend \
--pretrain "/opt_data/ecnu/resnet50.ckpt" \
--tag "sysu_all_part_graph" \
--data-path "" \
--loss-func "id+tri" \
--branch main \
--sysu-mode "all" \
--part 3 \
--graph \
--epoch 40