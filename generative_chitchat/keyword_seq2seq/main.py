#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf

from models import model_factory
from config import Config


def main(_):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=('train', 'test'), help='work mode')
    parser.add_argument('--model_dir', type=str, required=True, help='model directory')

    parser.add_argument('--config', type=str, help='config file containing parameters to configure the model')

    parser.add_argument('--pretrain_data', type=str, help='pretrain dataset')
    
    parser.add_argument('--train_data', type=str, help='training dataset')
    parser.add_argument('--dev_data', type=str, help='development dataset')
    parser.add_argument('--test_data', type=str, help='test dataset')

    parser.add_argument('--embed_conf', type=str, default="conf/word_embeddings.yml", help='embedding config file')
    parser.add_argument('--enable_epoch_evals', action='store_true',
                        help='enable evals after finishing an apoch during training')
    parser.add_argument('--enable_final_eval', action='store_true', help='enable the last eval once training finished')
    parser.add_argument('--disable_encoder_var_sharing', action='store_true',
                        help='disable encoders sharing variables to support testing old models')

    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--n_responses', type=int, default=1, help='number of generated responses')
    parser.add_argument('--beam_width', type=int, default=5, help='beam width to override the value in config file')
    parser.add_argument('--length_penalty_weight', type=float, default=0.8,
                        help='length penalty to override the value in config file')
    parser.add_argument('--sampling_temperature', type=float,
                        help='sampling temperature to override the value in config file')

    args = vars(parser.parse_args())
    config = Config(**args)

    model = model_factory.create_model(config)

    if config.mode == 'train':
        model.train()
    if config.mode == 'test':
        model.test()


if __name__ == "__main__":
    tf.app.run()