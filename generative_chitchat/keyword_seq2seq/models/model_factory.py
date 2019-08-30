#!/usr/bin/env python
# coding: utf-8


from models.topic_aware import taware_wrapper


def create_model(config):
    if config.type == 'topic_aware':
        return taware_wrapper.TopicAwareNMTEncoderDecoder(config)

    raise ValueError('unknown model: ' + config.type)
