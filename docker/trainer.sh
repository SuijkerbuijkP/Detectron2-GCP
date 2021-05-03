# please set variables that can vary between runs in the train_push_run file, as this file is read from a docker image

#OMP_NUM_THREADS=1 python trainer/run.py --config-file ./configs/mask_rcnn_R_101_FPN_3x.yaml --architecture "d2" --reuse-weights False --checkpoint 0009799 $@
OMP_NUM_THREADS=1 python trainer/run.py --config-file ./configs/R_101_dcni3_5x.yaml --architecture "adet" --reuse-weights False --checkpoint 0009799 --filter --categories "total_loss" "other" "dent" "shatter" $@ 
