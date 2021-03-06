#!/usr/bin/env bash

LOG_FILE="./log_k11.txt"
> $LOG_FILE
#for ((i=60000;i<=200000;i=i+10000)); do
#    echo -e "\n\n#### Evaluating model $i" 2>&1 >> $LOG_FILE
#    python eval_from_ckpt.py  \
#         --out_dir /ssd/wfei/results/wanda_b1may_lprresults_v2.2 \
#         --label_file /ssd/wfei/data/plate_for_label/wanda_b1_may/wanda_b1_may1011_ocrlabel.txt \
#         --model_ckpt ./model_wanda_fresh_0604/LPR_wanda.ckpt-${i}  2>&1 | tee -a $LOG_FILE ;
#done

#for ((i=60000;i<=185000;i=i+10000)); do
#    echo -e "\n\n#### Evaluating model $i" 2>&1 >> $LOG_FILE
#    python eval_from_ckpt.py  \
#         --out_dir /ssd/wfei/results/wanda_4k_lprresults_v2.2 \
#         --label_file /ssd/wfei/data/testing_data/wanda_benchmark_4k_ocrlabel.txt \
#         --model_ckpt ./model_wanda_fresh_0604/LPR_wanda.ckpt-${i}  2>&1 | tee -a $LOG_FILE ;
#done
CUDA_VISIBLE_DEVICES=1
for ((i=60000;i<=200000;i=i+10000)); do
    echo -e "\n\n#### Evaluating model $i" 2>&1 >> $LOG_FILE
    python eval_from_ckpt.py  \
         --out_dir /ssd/wfei/results/k11_4k_lprresults_v2.0 \
         --label_file /ssd/wfei/data/testing_data/k11_benchmark_ocrlabel.txt  \
         --model_ckpt ./model_k11/LPR_k11_withsyn.ckpt-${i}  2>&1 | tee -a $LOG_FILE ;
done
