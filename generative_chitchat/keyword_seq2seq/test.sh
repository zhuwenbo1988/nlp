export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64/
export CUDA_VISIBLE_DEVICES=""

DPATH='sample_data'

TEST_DATA=$DPATH'/test_data'
MODEL_DIR='log/1'

python -u main.py --mode test --model_dir $MODEL_DIR --test_data $TEST_DATA --beam_width 5 --length_penalty_weight 0.8