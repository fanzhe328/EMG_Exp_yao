# *-* coding=utf-8 *-*
# !/usr/bin/python
import numpy as np
import os
import sys
from file_script import log_result

root_path = os.getcwd()

def result_load(dir='250_100', feature_type='TD4', subject='subject_1', norm='_norm', action='7', training_type='intra'):
    file_path = root_path + '/result/' + dir + '/' +\
                feature_type+'_data4_'+ subject + norm + '/' +\
                'feat_'+feature_type+'_'+training_type+'_action_1-'+str(action)+'.npy'
    data = np.load(file_path)
    return data

def new_fold(log_fold):
    if os.path.isdir(log_fold) == False:
        try:
            os.makedirs(log_fold)
        except:
            print "Can not create log fold! "
            return False
    return True

def result_analyse():
    feature_type = 'TD4'
    norm = '_norm'
    training_type = 'intra'
    channel_pos_list = ['S0',                                             # 中心位置
                    'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右

    # action = 7
    # subject='subject_1'
    action_lists = [7, 9, 11]
    subject_list = ['subject_' + str(i) for i in range(1, 6)]
    
    for action in action_lists:
        for subject in subject_list:
            data = result_load('250_100',feature_type, subject, norm, action, training_type)
            
            title = feature_type+'_'+subject+'_action_1-'+str(action)
            res = []
            res_head = [title]
            res_head.extend(channel_pos_list[1:])
            res.append(res_head)
            
            index = 3
            span = len(channel_pos_list)-1

            res_intra = ['intra']
            res_intra.extend(data[index:index+span,4][:])
            res.append(res_intra)
            
            index += span
            res_center = ['center']
            res_center.extend(data[index:index+span,4][:])
            res.append(res_center)

            index += span
            res_group = ['group']
            res_group.extend(data[index:index+span,4][:])
            res.append(res_group)

            for i in range(6):
                n_components =  6+i*2
                index += span
                res_CCA = ['CCA_'+str(n_components)]
                res_CCA.extend(data[index:index+span,4][:])
                res.append(res_CCA)

            fold_path = root_path + '/result/cca_analyse'
            new_fold(fold_path)

            file_path = fold_path + '/' + title
            log_result(res, file_path, 2)
    # print channel_pos_list, channel_pos_list[:]

if __name__ == '__main__':
    result_analyse()