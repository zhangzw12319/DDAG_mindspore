export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

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
--gpu 3 \
--device-target GPU \
--pretrain "resnet50.ckpt" \
--tag "sysu_indoor_baseline" \
--data-path "/home/shz/pytorch/data/sysu" \
--loss-func "id+tri" \
--branch main \
--sysu-mode "indoor" \
--part 0