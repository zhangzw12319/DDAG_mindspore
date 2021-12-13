export PATH=/usr/local/cuda-10.1/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib/:$LD_LIBRARY_PATH

myfile="test_regdb_i2v.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_eval and run. Exit..."
    exit 0
fi

cd ../..

python eval.py \
--dataset RegDB \
--device-id 0 \
--device-target Ascend \
--resume "XXX.ckpt" \
--tag "regdb_i2v" \
--data-path "Define your own path/regdb" \
--branch main \
--regdb-mode "i2v" \
--part 0