# *-* coding: UTF-8 *-*
import os
import sys
import numpy as np
import data_load
import matplotlib.pyplot as plt
from preprocess import load_raw_dataset
from file_script import new_fold
root_path = os.getcwd()


def feature_action_sensitivity(feature_type='TD4'):
    ''' 对每个特征，分析其在不移位和移位情况下的差异性 '''
    results = []
    
    subjects = ['subject_' + str(i + 1) for i in range(5)]

    channel_pos_list = ['S0',                                             # 中心位置
                        'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右
    pos_num = len(channel_pos_list)
    
    actions = [i+1 for i in range(7)]
    action_num = len(actions)                        # 7 动作类型个数

    if feature_type == 'TD4':
        feature_list = ['MAV', 'ZC', 'SSC', 'WL']
    elif feature_type == 'TD5':
        feature_list = ['MAV', 'ZC', 'SSC', 'WL','RMS']
    feat_num = len(feature_list)                    # 4 特征维度

    groups = [i+1 for i in range(4)]
    group_num = len(groups)                         # 4 通道数

    action_span = feat_num*group_num                # 16
    # print groups, channel_num, channel_span, feat_num
    
    train_dir = 'train4_250_100'


    results.append(['subject', 'action', 'feature', 'group','means_shift', 'std_shift'] )

    for subject in subjects:
        shift_simulation = np.ones((action_num,action_span,2))
        trains, classes = data_load.load_feature_dataset(train_dir, subject, feature_type)
        gaussion_distribute = np.ones( (len(actions), len(groups), len(feature_list), 2))
        for action in actions:
            trains_action = trains[classes == action]
            means = np.mean(trains_action, axis=0)
            stds = np.std(trains_action, axis=0)
            
            for group in groups:
                for feat_idx, feat_name in enumerate(feature_list):
                    idx_S0 = (group-1)*feat_num+feat_idx
                    idx = np.array([(i+1)*action_span+feat_idx+(group-1)*feat_num 
                            for i in range(pos_num-1)])
            
                    # means_shift = abs(means[idx] - means[idx_S0])/means[idx_S0]
                    # means_shift = abs(means[idx] - means[idx_S0])/means[idx_S0] \
                    #             + abs(stds[idx]-stds[idx_S0])/stds[idx_S0]
                    # results.append([subject, str(action), feat_name, str(group)] + map(str, means_shift))
                    
                    means_shift = np.mean(means[idx]) - np.mean(means[idx_S0])
                    std_shift = np.std(stds[idx]) - np.std(stds[idx_S0])
                    shift_simulation[action-1, (group-1)*feat_num+feat_idx, :] = np.array([means_shift, std_shift]) 
                    # means_shift = abs(means[idx] - means[idx_S0])/means[idx_S0] \
                    #             + abs(stds[idx]-stds[idx_S0])/stds[idx_S0]
                    results.append([subject, str(action), feat_name, str(group), str(means_shift), str(std_shift)])
                    gaussion_distribute[action-1, group-1, feat_idx,:] = [means_shift, std_shift]
                    # print subject, action, feat_name, group, means_shift[:]
        log_result(gaussion_distribute, root_path + '/result/sensitivity/'+subject+'_simulation_1', 2)       
    log_result(results, root_path + '/result/sensitivity/feature_action_sensitivity_5', 2)


def feature_action_sensitivity(featuret_type='TD4', action_num=7):
    ''' 对每个特征，分析其在不移位和移位情况下的差异性 '''

    results = []
    subjects = ['subject_' + str(i + 1) for i in range(5)]
    channel_pos_list = ['S0',                                             # 中心位置
                        'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右
    pos_num = len(channel_pos_list)
    
    actions = [i+1 for i in range(action_num)]
    groups = [i+1 for i in range(4)]
    if feature_type == 'TD4':
        feature_list = ['MAV', 'ZC', 'SSC', 'WL']
    elif feature_type == 'TD5':
        feature_list = ['MAV', 'ZC', 'SSC', 'WL','RMS']
    feat_num = len(feature_list)                    # 4 特征维度

    groups = [i+1 for i in range(4)]
    group_num = len(groups)                         # 4 通道数

    action_span = feat_num*group_num                # 16

    train_dir = 'train4_250_100'


    results.append(['subject', 'action', 'feature', 'group'] + channel_pos_list[1:])
    for subject in subjects:
        for action in actions:
            filename = subject+'_feat_'+featuret_type+'_action_'+str(action)
            data = np.load(root_path + '/train4_250_100/' + filename + '.npy')
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)

            for group in groups:
                for feat_idx, feat_name in enumerate(feature_list):
                    idx = np.array([(i+1)*action_span+feat_idx+(group-1)*feat_num 
                            for i in range(pos_num-1)])
                    idx_S0 = (group-1)*feat_num+feat_idx
                    # print idx, idx_S0
                    # variation = abs(means[idx] - means[idx_S0])/means[idx_S0] \
                    #             + abs(stds[idx]-stds[idx_S0])/stds[idx_S0]
                    variation = abs(means[idx] - means[idx_S0])/means[idx_S0]
                    results.append([subject, str(action), feat_name, str(group)] + map(str, variation))
                    # print results
                    # print subject, action, feat_name, group, variation[:]
    fold_path = root_path + '/result_gaussian/sensitivity/'
    new_fold(fold_path) 
    log_result(results, fold_path+'feature_action_sensitivity', 2)

def log_result(results, log_file, flag):
    np.save(log_file + '.npy', results)
    if flag == 2:
        np.savetxt(log_file + '.csv', results, fmt="%s", delimiter=",")


if __name__ == '__main__':
    feature_type = 'TD4'
    feature_action_sensitivity(feature_type)