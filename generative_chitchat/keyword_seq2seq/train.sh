export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64/
export CUDA_VISIBLE_DEVICES=""

DPATH='sample_data'

YAML_FILE='conf/taware_small.yml'
TRAIN_DATA=$DPATH'/train_data'
DEV_DATA=$DPATH'/dev_data'
TEST_DATA=$DPATH'/test_data'
MODEL_DIR='log/1'

python -u main.py --mode train \
--config $YAML_FILE \
--train_data $TRAIN_DATA \
--dev_data $DEV_DATA \
--test_data $TEST_DATA \
--num_gpus 1 \
--model_dir $MODEL_DIR
