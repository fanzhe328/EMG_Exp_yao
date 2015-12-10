# *-* coding=utf-8 *-*
# !/usr/bin/python
import numpy as np
import os, sys, time
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

def result_integrate_intra(time_now):
    training_type = 'intra'
    feature_type = 'TD4'
    norm = '_norm'
    channel_pos_list = ['S0',                                             # 中心位置
                    'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右

    # action = 7
    # subject='subject_1'
    action_lists = [7, 9, 11]
    subject_list = ['subject_' + str(i) for i in range(1, 6)]
    res_all = []
    blank_line = ['' for i in range(len(channel_pos_list)+1)]
    for i in range(5):
        res_all.append(blank_line)

    
    fold_path = root_path + '/result/cca_analyse'
    new_fold(fold_path)
    
    for action in action_lists:
        for subject in subject_list:
            data = result_load('250_100',feature_type, subject, norm, action, training_type)
            
            title = feature_type+'_'+subject+'_action_1-'+str(action)
            res = []
            res_head = [title]
            res_head.extend(channel_pos_list[1:])
            res_head.append('Average')
            res.append(res_head)
            
            index = 3
            span = len(channel_pos_list)-1

            res_intra = ['intra']
            temp = map(float, data[index:index+span,4][:])
            res_intra.extend(temp)
            res_intra.append(np.mean(temp))
            res.append(res_intra)
            
            index += span
            res_center = ['center']
            temp = map(float, data[index:index+span,4][:])
            res_center.extend(temp)
            res_center.append(np.mean(temp))
            res.append(res_center)

            index += span
            res_group = ['group']
            temp = map(float, data[index:index+span,4][:])
            res_group.extend(temp)
            res_group.append(np.mean(temp))
            res.append(res_group)

            for i in range(6):
                n_components =  6+i*2
                index += span
                res_CCA = ['CCA_'+str(n_components)]
                temp = map(float, data[index:index+span,4][:])
                res_CCA.extend(temp)
                res_CCA.append(np.mean(temp))
                res.append(res_CCA)

            res_all.extend(res)
            for j in range(10):
                res_all.append(blank_line)

    file_path = fold_path + '/' + feature_type+'_'+training_type+'_'+time_now
    log_result(res_all, file_path, 2)
    # print channel_pos_list, channel_pos_list[:]

if __name__ == '__main__':
    time_now = time.strftime('%Y-%m-%d_%H-%M',time.localtime(time.time()))
    result_integrate_intra(time_now)