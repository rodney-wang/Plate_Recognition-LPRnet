#!/usr/bin/env bash

LOG_FILE="./log.txt"
for ((i=100000;i<=200000;i=i+8000)); do
    echo -e "\n\n#### Evaluating model $i" >> $LOG_FILE
    python eval_from_ckpt.py  \
         --img_dir /ssd/wfei/data/plate_for_label/wanda_b1_may/plates \
         --out_dir /ssd/wfei/results/wanda_b1may_lprresults_v2.2 \
         --label_file /ssd/wfei/data/plate_for_label/wanda_b1_may/wanda_b1_may1011_ocrlabel.txt \
         --model_ckpt ./model_wanda_0604/LPR_wanda.ckpt-${i} 2>\$1 | tee -a $LOG_FILE
done