export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

myfile="test_regdb_v2i.sh"

if [! -f "$myfile"]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python eval.py \
--dataset RegDB \
--gpu 2 \
--device-target GPU \
--resume "logs/regdb_v2i/training/epoch_35_rank1_85.49_mAP_80.01_RegDB_batch-size_2*8*4=64_adam_lr_0.0035_loss-func_id+tri_Graph__trial_1_main.ckpt" \
--tag "regdb_v2i" \
--data-path "/home/shz/pytorch/data/regdb" \
--branch main \
--regdb-mode "v2i" \
--part 0