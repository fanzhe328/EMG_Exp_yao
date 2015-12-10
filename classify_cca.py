# *-* coding=utf-8 *-*
# !/usr/bin/python
'''
	classify_all_channel
	所有通道全部作为训练数据和测试数据
'''

import numpy as np
import os
import sys
import time
import data_load
import classifier_lda_cca
from preprocess import data_preprocess, data_normalize
from result_integrate import result_integrate_intra

# from noise_simulation import guassion_simu
# from data_plot import plot_result


root_path = os.getcwd()


def train_dataset_feature_inter(train_dir='train4_250_100', subject_list=['subject_1'], 
    feature_type='TD4', dataset='data4',fold_pre='250_100', z_score=False, 
    channel_pos_list=['S0'], action_num=7, chan_num=4):
    my_clfs = ["LDA"]

    start_time = time.time()

    channel_pos_list_shift = channel_pos_list[1:]

    if feature_type == 'TD4':
        feat_num = 4                    # 特征维度 TD4:4
    elif feature_type == 'TD5':
        feat_num = 5

    chan_len = feat_num * chan_num             # 16
    norm = ''

    for sub in subject_list:
        trains, classes = data_load.load_feature_dataset(
            train_dir, sub, feature_type, action_num)
        
        if z_score:
            trains = data_normalize(trains)
            norm = '_norm'

        trains_S0 = trains[:, 0:chan_len]
        trains_shift = trains[:, chan_len:]

        num = 1

        # 生成CCA映射矩阵
        # classifier_lda_cca.generate_transform_equations(
        #     trains_S0, trains_shift, pos_list=channel_pos_list_shift, chan_len=chan_len, subject=sub)
        
        # 进行中心训练策略
        classifier_lda_cca.training_lda_TD4_inter(
            my_clfs, trains_S0, trains_shift, classes,
            log_fold=fold_pre + '/' + feature_type + '_' + dataset + '_' + sub + norm,
            pos_list=channel_pos_list_shift, chan_len=chan_len, chan_num=chan_num,
            feature_type=feature_type, action_num=action_num, num=num, subject=sub)

        print "Total times: ", time.time() - start_time, 's'


def train_dataset_feature_intra( train_dir='train1', subject_list=['subject_1'], 
    feature_type='TD4', dataset='data1', fold_pre='250_100', z_score=False, 
    channel_pos_list=['O'], action_num=7, chan_num=4):
    
    print 'train_dataset_feature_intra......'

    my_clfs = ["LDA"]
    start_time = time.time()

    if feature_type == 'TD4':
        feat_num = 4
    if feature_type == 'TD5':
        feat_num = 5
    chan_len = feat_num * chan_num

    # 0均值标准化标志
    norm = ''
    for sub in subject_list:
        trains, classes = data_load.load_feature_dataset(
            train_dir, sub, feature_type, action_num)

        # 是否进行0均值标准化
        if z_score:
            trains = data_normalize(trains)
            norm = '_norm'

        classifier_lda_cca.training_lda_TD4_intra(
            my_clfs, trains, classes,
            log_fold=fold_pre + '/' + feature_type + '_' + dataset + '_' + sub + norm,
            pos_list=channel_pos_list, num=1, chan_len=chan_len, action_num=action_num,
            feature_type=feature_type, chan_num=chan_num)

    print "Total times: ", time.time() - start_time, 's'

if __name__ == '__main__':
    winsize = 250
    incsize = 100
    samrate = 1024
    fold_pre = str(winsize) + '_' + str(incsize)

    feature_type = 'TD4'
    # feature_type = 'TD5'

    actions = [7, 9, 11]
    # actions = [7]

    train_dir = 'train4_' + fold_pre
    input_dir = 'data4'
    chan_num = 4
    subject_list = ['subject_' + str(i) for i in range(1, 6)]

    channel_pos_list = ['S0',                                             # 中心位置
                        'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右

    z_scores = [True]

    for z_score in z_scores:
        for action_num in actions:
            train_dataset_feature_intra(train_dir, subject_list, feature_type,
                input_dir, fold_pre, z_score, channel_pos_list, action_num)

            # train_dataset_feature_inter(train_dir, subject_list, feature_type,
            #     input_dir, fold_pre, z_score,channel_pos_list, action_num)
    
    time_now = time.strftime('%Y-%m-%d_%H-%M',time.localtime(time.time()))
    result_integrate_intra(time_now)
