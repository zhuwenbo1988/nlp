#!/bin/bash
export LD_LIBRARY_PATH=/home/cuda/cuda-9.0/lib64/:/home/cuda/cuda-10.1/extras/CUPTI/lib64/
export CUDA_VISIBLE_DEVICES="6"

set -e

# 定义任务的名称
export NAME_SPACE=e2e_sample
# 定义bert model 路径
export BERT_BASE_DIR=./models/chinese_L-12_H-768_A-12
# 定义bert 最大长度
export MAX_SEQ_LENGTH=32
# 定义bert batch size 
export TRAIN_BATCH_SIZE=16
# 定义bert learing rate
export LEARNING_RATE=5e-5
# 定义bert epochs
export NUM_TRAIN_EPOCHS=40
# 定义只使用crf不要bi lstm
export CRF_ONLY=False

# 默认设置，最好不要改
export OUTPUT_DIR=${NAME_SPACE}/model_output

rm -rf ${NAME_SPACE}

mkdir ${NAME_SPACE}
mkdir -p ${OUTPUT_DIR}

cp ./e2e_lu_sample_data/with_lexicon/*_labels.txt ${NAME_SPACE}
cp ./e2e_lu_sample_data/with_lexicon/*_data.json ${NAME_SPACE}
cp ./e2e_lu_sample_data/with_lexicon/domain_intent_lexicon_names.txt ${NAME_SPACE}
cp ./e2e_lu_sample_data/with_lexicon/slot_lexicon_names.txt ${NAME_SPACE}

python bert_e2e_lu_lexicon.py   \
      --task_name=$NAME_SPACE  \
      --do_train=True   \
      --do_eval=False  \
      --do_predict=True \
      --do_export=False \
      --use_domain_intent_lexicon=True \
      --use_slot_lexicon=True \
      --use_domain_vector=False \
      --data_dir=$NAME_SPACE   \
      --vocab_file=$BERT_BASE_DIR/vocab.txt  \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   \
      --max_seq_length=$MAX_SEQ_LENGTH   \
      --train_batch_size=$TRAIN_BATCH_SIZE   \
      --learning_rate=$LEARNING_RATE   \
      --num_train_epochs=$NUM_TRAIN_EPOCHS   \
      --crf_only=${CRF_ONLY} \
      --output_dir=$OUTPUT_DIR
