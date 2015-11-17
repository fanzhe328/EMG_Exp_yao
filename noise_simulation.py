'''
Author: Fanz
'''

import numpy as np
# from data_load import load_dataset1
import os, sys, random

root_path = os.getcwd()

def proportion_simu(train, target, proportion=1.0):
    train_porportion = train * proportion
    train_simu = np.concatenate((train, train_porportion), axis=0)
    target_simu = np.concatenate((target, target), axis=0)
    return train_simu, target_simu

def load_gaussion_distribute(subject='subject_1'):
    file_path = root_path + '/result/sensitivity/' + subject + '_simulation_1.npy'
    return np.load(file_path)

def guassion_simu(trains, targets, subject, action_num, chan_num, feat_num):
    # gaussion_distribute    (actions, groups, feat_num, mean+std)
    subject = 'subject_1'
    gd = load_gaussion_distribute(subject)
    simu_times = 10
    trains_simu = np.ones( (trains.shape[0]*simu_times, trains.shape[1]))
    targets_simu = np.ones( (targets.shape[0]*simu_times,), np.int)
    actions = [i+1 for i in range(action_num)]
    channels = [i+1 for i in range(chan_num)]
    feats  = [i+1 for i in range(feat_num)]
    start = 0
    for action in actions:
        trains_temp = trains[targets==action]
        temp_len = trains_temp.shape[0]
        targets_temp = np.ones( (temp_len,),np.int)*action
        
        means = np.mean(trains_temp, axis=0)
        stds = np.std(trains_temp, axis=0)

        trains_simu[start:start+temp_len,:] = trains_temp
        targets_simu[start:start+temp_len] = targets_temp
        
        for i in range(simu_times-1):
            start += temp_len
            targets_simu[start:start+temp_len] = targets_temp
            for i in range(temp_len):
                for c in range(chan_num):
                    for f in range(feat_num):
                        # print means[f+c*feat_num], gd[action-1, c, f, 0],random.uniform(0,gd[action-1, c, f, 0]), gd[action-1, c, f, 1],random.uniform(-gd[action-1, c, f, 1], gd[action-1, c, f, 1])
                        # sys.exit(0)
                        trains_temp[i, f+c*feat_num] =\
                            means[f+c*feat_num] + random.uniform(0,gd[action-1, c, f, 0])/2\
                            + random.uniform(-gd[action-1, c, f, 1], gd[action-1, c, f, 1])
            trains_simu[start:start+temp_len,:] = trains_temp

        start += temp_len   

    # print actions, channels, feats
    # sys.exit(0)
    # print trains.shape, targets.shape
    # print trains_simu.shape, targets_simu.shape
    return trains_simu, targets_simu

if __name__ == '__main__':
    print __doc__
    # gd = load_gaussion_distribute()
    # print gd.shape

    # train, target = load_dataset1()
    # train_simu, target_simu = proportion_simu(train[1], target, 0.9)
    # print train_simu.shape, target_simu.shape
    # iteration = 10
    # while(iteration > 0):
    #     print random.randrange(1, 51)
    #     print random.randrange(1, 51)
    #     print random.randrange(1, 51)
    #     print random.randrange(1, 51)
    #     print random.randrange(1, 51)
    #     print random.randrange(1, 51)
    #     iteration -= 1
