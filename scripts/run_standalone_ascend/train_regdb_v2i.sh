myfile="train_regdb_v2i.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python train.py \
--dataset RegDB \
--optim adam \
--lr 0.0035 \
--device-id 1 \
--device-target Ascend \
--pretrain "resnet50.ckpt" \
--tag "regdb_v2i" \
--data-path "" \
--loss-func "id+tri" \
--branch main \
--regdb-mode "v2i" \
--part 0 \
--graph