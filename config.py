#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ config.py
 Author @ huangjunheng
 Create date @ 2018-06-17 16:09:27
 Description @ config
"""


class Config(object):
    """
    模型配置
    """
    # 定义训练参数
    learning_rate = 0.01
    training_epoch = 2
    batch_size = 100
    training_steps = 1000

    # 定义隐藏层数目
    num_hidden = 128

    # 定义层数
    num_layers = 1

    # 数据位置
    training_file = 'data/v1/training_test_data/training_data.txt'
    test_file = 'data/v1/training_test_data/test_data.txt'
    predict_file = 'data/training_test_data/predict_data.txt'

    # model path
    save_model_path = "model/v1/train_model.ckpt"
    load_model_path = "model/v1/train_model.ckpt"