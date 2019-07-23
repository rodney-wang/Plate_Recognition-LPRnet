#!/usr/bin/env bash

LOG_FILE="./log_k11_bms.txt"
> $LOG_FILE

MODEL=./model_pb/model_k11_pb_0720
#test the 4k benchmark
python eval_lpr.py --ocr_txt /ssd/wfei/data/testing_data/k11_benchmark_ocrlabel.txt  \
                   --out_dir /ssd/wfei/results/k11_lprresults \
                   --model $MODEL 2>&1 | tee -a $LOG_FILE

#test the hongkong plate
python eval_lpr.py --ocr_txt /ssd/wfei/data/plate_for_label/hk_double/hkdouble_0505_ocrlabel.txt  \
                   --out_dir /ssd/wfei/results/hkdouble_lprresult \
                   --model $MODEL 2>&1 | tee -a $LOG_FILE

#test the new energy plate
python eval_lpr.py  --ocr_txt  /ssd/wfei/data/plate_for_label/energy_cars/energy_plates_ocrlabel_test_217.txt  \
                    --out_dir /ssd/wfei/results/k11_energy_lprresults  \
                    --model $MODEL 2>&1 | tee -a $LOG_FILE


#test the entrance plate
python eval_lpr.py  --ocr_txt  /ssd/wfei/data/plate_for_label/k11_entrance/20190424_k11_entrance_ocrlabel.txt  \
                    --out_dir /ssd/wfei/results/k11_entrance_lprresults  \
                    --model $MODEL 2>&1 | tee -a $LOG_FILE


#test the difficult plate
python eval_lpr.py  --ocr_txt  /ssd/wfei/data/plate_for_label/k11_difficult/k11_difficult_test_ocrlabel.txt  \
                    --out_dir /ssd/wfei/results/k11_difficult_lprresults  \
                    --model $MODEL 2>&1 | tee -a $LOG_FILE