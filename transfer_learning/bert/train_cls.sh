#!/bin/bash
export LD_LIBRARY_PATH=/home/cuda/cuda-9.0/lib64/:/home/cuda/cuda-10.1/extras/CUPTI/lib64/
export CUDA_VISIBLE_DEVICES="6"

set -e

# 定义任务的名称
export NAME_SPACE=sentiment_cls_3
# 定义bert model 路径
export BERT_BASE_DIR=./models/uncased_L-12_H-768_A-12
# 定义bert 最大长度
export MAX_SEQ_LENGTH=128
# 定义bert batch size 
export TRAIN_BATCH_SIZE=32
# 定义bert learing rate
export LEARNING_RATE=5e-5
# 定义bert epochs
export NUM_TRAIN_EPOCHS=10

export OUTPUT_DIR=${NAME_SPACE}/model_output
rm -rf ${NAME_SPACE}
mkdir ${NAME_SPACE}
mkdir -p ${OUTPUT_DIR}
cp ./semeval_data/labels.txt ${NAME_SPACE}
cp ./semeval_data/*_data.json ${NAME_SPACE}
cp ./semeval_data/review.json ${NAME_SPACE}

python run_classifier.py   \
      --task_name=$NAME_SPACE  \
      --do_train=False   \
      --do_eval=False  \
      --do_predict=True \
      --data_dir=$NAME_SPACE   \
      --vocab_file=$BERT_BASE_DIR/vocab.txt  \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   \
      --max_seq_length=$MAX_SEQ_LENGTH   \
      --train_batch_size=$TRAIN_BATCH_SIZE   \
      --learning_rate=$LEARNING_RATE   \
      --num_train_epochs=$NUM_TRAIN_EPOCHS   \
      --output_dir=$OUTPUT_DIR