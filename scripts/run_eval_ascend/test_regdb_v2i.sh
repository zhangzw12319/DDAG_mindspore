#!/usr/bin/env bash
myfile="test_regdb_v2i.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python eval.py \
--dataset RegDB \
--device-id 0 \
--device-target Ascend \
--resume "XXX.ckpt" \
--tag "regdb_v2i" \
--data-path "Define your own path/regdb" \
--branch main \
--regdb-mode "v2i" \
--part 0