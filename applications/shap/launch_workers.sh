#!/bin/bash


for work in {0..9}
do
    sed "s/worker=0/worker=${work}/" pbs_launch.sh > tmp.sh
    mv tmp.sh launch.sh
    qsub -N "shap-worker-${work}" launch.sh
done

rm launch.sh