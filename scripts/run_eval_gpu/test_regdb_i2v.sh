#!/usr/bin/env bash
myfile="test_regdb_i2v.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_eval and run. Exit..."
    exit 0
fi

cd ../..

python eval.py \
--dataset RegDB \
--gpu 0 \
--device-target GPU \
--resume "XXX.ckpt" \
--tag "regdb_i2v" \
--data-path "Define your own path/regdb" \
--branch main \
--regdb-mode "i2v" \
--part 0