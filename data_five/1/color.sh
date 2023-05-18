#!/bin/bash
TXT_PATH="/home/s172096/research/bert/japanese-emphasize/data_five/1/test.txt"
EMP_PATH="/home/s172096/t-test/201203/BERT-CRF/with_crf_1_threshold.txt"
paste <(cat $TXT_PATH |cut -d " " -f 1) <(paste <(cat $TXT_PATH |cut -d " " -f 1) <(cat $EMP_PATH |grep '^[01]*$')|awk '{print $2}')|awk '{if($2=="1"){printf "\033[32m"$1"\033[0m"" "}else{printf $1" "}}'

