export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

myfile="train_sysu_all_part_graph.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python train.py \
--MSmode "PYNATIVE_MODE" \
--dataset SYSU \
--optim adam \
--lr 0.0035 \
--gpu 3 \
--device-target GPU \
--pretrain "resnet50.ckpt" \
--tag "sysu_all_part_graph" \
--data-path "/home/shz/pytorch/data/sysu" \
--loss-func "id+tri" \
--branch main \
--sysu-mode "all" \
--part 3 \
--graph \
--epoch 40