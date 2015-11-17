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
import classifier_lda
from preprocess import data_preprocess, data_normalize
from noise_simulation import guassion_simu
# from data_plot import plot_result


root_path = os.getcwd()


def train_dataset_feature_inter(
        train_dir='train4_250_100', subject_list=['subject_1'], feature_type='TD4', dataset='data4',
        fold_pre='250_100', z_score=False, channel_pos_list=['S0'], action_num=7, group_num=4):
    my_clfs = ["LDA"]

    start_time = time.time()
    # channel_pos_list_shift = channel_pos_list[1:]
    channel_pos_list_shift = channel_pos_list
    action_num = 7
    group_num = 4
    if feature_type == 'TD4':
        feat_num = 4                    # 特征维度 TD4:4 
    elif feature_type == 'TD5':
        feat_num = 5

    chan_num = 4                        # 通道个数，4通道
    chan_len = feat_num * chan_num             # 16

    for sub in subject_list:
        trains, classes = data_load.load_feature_dataset(train_dir, sub, feature_type)

        tests_inter = np.array([])
        trains_inter = trains[:, 0:chan_len]
        trains_simu, classes_simu = guassion_simu(trains_inter, classes, sub, action_num, chan_num, feat_num)
        # trains_simu, classes_simu = trains_inter, classes

        tests_inter = trains
        
        if z_score:
            trains_simu = data_normalize(trains_simu)
            tests_inter = data_normalize(tests_inter)
            sub = 'norm_' + sub

        # print channel_pos_list
        # print trains_inter.shape, tests_inter.shape
        # sys.exit(0)
# 
        num = 5

        classifier_lda.training_lda_TD4_inter(
            my_clfs, trains_simu, classes_simu, tests_inter, classes,
            log_fold=fold_pre + '/' + feature_type + '_' + dataset + '_' + sub + '_simu1',
            pos_list=channel_pos_list_shift, chan_len=chan_len, group_num=group_num,
            feature_type=feature_type, action_num=action_num, num=num)
        print "Total times: ", time.time() - start_time, 's'


def train_dataset_feature_intra(
        train_dir='train1', subject_list=['subject_1'], feature_type='TD4', dataset='data1',
        fold_pre='250_100', z_score=False, channel_pos_list=['O'], action_num=7, group_num=4):
    my_clfs = ["LDA"]
    start_time = time.time()
    for sub in subject_list:
        trains, classes = data_load.load_feature_dataset(train_dir, sub, feature_type)
        chan_num = 4                        # 通道个数，4通道
        # if z_score:
        #     trains = data_normalize(trains)
        #     sub = 'norm_' + sub
        if feature_type == 'TD4':
            feat_num = 4
        if feature_type == 'TD5':
            feat_num = 5

        chan_len = feat_num * chan_num

        classifier_lda.training_lda_TD4_intra(
            my_clfs, trains, classes,
            log_fold=fold_pre + '/' + feature_type + '_' + dataset + '_' + sub + '_simu1',
            pos_list=channel_pos_list, num=1, chan_len=chan_len,action_num=action_num,
            feature_type=feature_type,group_num=group_num)
    print "Total times: ", time.time() - start_time, 's'


def train_dataset_signal():
    my_clfs = ["LDA"]
    # subject = 'subject_1'
    subject_list = ['subject_' + str(i) for i in range(1, 2)]
    # trains, classes = load_feature_dataset('train1', 'subject_1', feature_type)

    # trains1, classes1, trains2, classes2 = load_feature_dataset('train4', subject)
    # print trains.shape, classes.shape

    start_time = time.time()

    for sub in subject_list:
        trains, classes = load_signal_dataset('train1', sub)
        print trains.shape, classes.shape
        classifier.training_lda_signal(
            my_clfs, trains, classes, log_fold='signal_data1_' + sub, num=1)

    # classifier.training_lda_TD4_cross(my_clfs, trains1, classes1, trains2, classes2, log_fold = 'TD4_data4_'+subject+'_1to2', num=1)
    # classifier.training_lda_TD4_cross(my_clfs, trains2, classes2, trains1, classes1, log_fold = 'TD4_data4_'+subject+'_2to1', num=1)
    print "Total times: ", time.time() - start_time, 's'


def main():
    pass

if __name__ == '__main__':
    winsize = 250
    incsize = 100
    samrate = 1024
    fold_pre = str(winsize) + '_' + str(incsize)
    
    feature_type = 'TD4'
    # feature_type = 'TD5'
    
    z_score = False
    action_num = 7

    train_dir = 'train4_' + fold_pre
    input_dir = 'data4'
    chan_num = 4
    subject_list = ['subject_' + str(i) for i in range(1, 2)]
    # channel_pos_list = ['O',								# 中心位置
    #                     'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1',		# 八方位 1cm 模拟：右，右下，下，左下，左，左上，上，右上
    #                     'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2']		# 八方位 2cm 模拟：同上
    channel_pos_list = ['S0',                                             # 中心位置
                        'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右
    # channel_pos_list = ['O']

    # data_preprocess(
    #     input_dir, train_dir, feature_type,
    #     subject_list, winsize, incsize, samrate)

    z_scores = [True]

    # for z_score in z_scores:
    #     train_dataset_feature_intra(
    #         train_dir, subject_list, feature_type,
    #         input_dir, fold_pre, z_score, 
    #         channel_pos_list,action_num)

    for z_score in z_scores:
        train_dataset_feature_inter(
            train_dir, subject_list, feature_type,
            input_dir, fold_pre, z_score, 
            channel_pos_list,action_num)

    # train_dataset_feature(train_dir, subject_list,
    #                       feature_type, input_dir, fold_pre, z_score)
    # train_dataset_signal()
