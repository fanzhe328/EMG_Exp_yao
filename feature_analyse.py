# *-* coding: UTF-8 *-*
# !/usr/bin/python

import os, sys
import numpy as np
import data_load
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSCanonical
root_path = os.getcwd()

def feature_action_sensitivity(feature_type='TD4'):
    ''' 对每个特征，分析其在不移位和移位情况下的协方差 '''
    results = []
    
    subjects = ['subject_' + str(i + 1) for i in range(1)]

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
    group_span = group_num*feat_num
    # print group_span
    action_span = feat_num*group_num                # 16
    # print groups, channel_num, channel_span, feat_num
    
    train_dir = 'train4_250_100'


    results.append(['subject', 'action', 'feature', 'group', 'means_shift', 'std_shift'] )
    plsca = PLSCanonical(n_components=2)
    # pos = 1
    k=0
    for pos_idx, pos_name in enumerate(channel_pos_list[1:]):
        pos = pos_idx+1
        for subject in subjects:
            # shift_simulation = np.ones((action_num,action_span,2))
            trains, classes = data_load.load_feature_dataset(train_dir, subject, feature_type)
            # m = trains.shape[0]
            # print trains.shape, classes.shape, m
            # print group_span, group_span*2
            # sys.exit(0)
            # m = trains.shape[0]*2/3
            m = trains.shape[0]/2
            X_train = trains[:m, group_span*pos: group_span*(pos+1)]
            Y_train = trains[:m:, :group_span]
            X_test = trains[m:, group_span*pos: group_span*(pos+1)]
            Y_test = trains[m:, :group_span]

            plsca.fit(X_train, Y_train)
            X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
            X_test_r, Y_test_r = plsca.transform(X_test, Y_test)

            filename=subject+'_'+pos_name
            # plot_plsc_figure(X_train_r,Y_train_r,X_test_r, Y_test_r, filename)
            plot_plsc_figure_two(X_train_r,Y_train_r,X_test_r, Y_test_r, filename)
            # if subject == "subject_1":
            #     X_train = trains[:, group_span*pos: group_span*(pos+1)]
            #     Y_train = trains[:, :group_span]
            #     plsca.fit(X_train, Y_train)

            # elif subject != "subject_1":
            #     X_test = trains[:, group_span*pos: group_span*(pos+1)]
            #     Y_test = trains[:, :group_span]

            #     X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
            #     X_test_r, Y_test_r = plsca.transform(X_test, Y_test)
            #     filename='train_subject_1_test_'+subject+'_'+pos_name
            #     plot_plsc_figure(X_train_r,Y_train_r,X_test_r, Y_test_r, filename)

def plot_plsc_figure(X_train_r,Y_train_r,X_test_r, Y_test_r, filename):
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.plot(X_train_r[:, 0], Y_train_r[:, 0], "ob", label="train")
    plt.plot(X_test_r[:, 0], Y_test_r[:, 0], "or", label="test")
    plt.xlabel("x scores")
    plt.ylabel("y scores")
    plt.title('Dimensino 1: X vs Y (test corr = %.2f)' %
              np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1])
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc="best")

    plt.subplot(224)
    plt.plot(X_train_r[:, 1], Y_train_r[:, 1], "ob", label="train")
    plt.plot(X_test_r[:, 1], Y_test_r[:, 1], "or", label="test")
    plt.xlabel("x scores")
    plt.ylabel("y scores")
    plt.title('Comp. 2: X vs Y (test corr = %.2f)' %
              np.corrcoef(X_test_r[:, 1], Y_test_r[:, 1])[0, 1])
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc="best")

    # 2) Off diagonal plot components 1 vs 2 for X and Y
    plt.subplot(222)
    plt.plot(X_train_r[:, 0], X_train_r[:, 1], "*b", label="train")
    plt.plot(X_test_r[:, 0], X_test_r[:, 1], "*r", label="test")
    plt.xlabel("X comp. 1")
    plt.ylabel("X comp. 2")
    plt.title('X comp. 1 vs X comp. 2 (test corr = %.2f)'
              % np.corrcoef(X_test_r[:, 0], X_test_r[:, 1])[0, 1])
    plt.legend(loc="best")
    plt.xticks(())
    plt.yticks(())

    plt.subplot(223)
    plt.plot(Y_train_r[:, 0], Y_train_r[:, 1], "*b", label="train")
    plt.plot(Y_test_r[:, 0], Y_test_r[:, 1], "*r", label="test")
    plt.xlabel("Y comp. 1")
    plt.ylabel("Y comp. 2")
    plt.title('Y comp. 1 vs Y comp. 2 , (test corr = %.2f)'
              % np.corrcoef(Y_test_r[:, 0], Y_test_r[:, 1])[0, 1])
    plt.legend(loc="best")
    plt.xticks(())
    plt.yticks(())
    # print subject
    plt.savefig(root_path + "/result/figure/cca/" + filename + ".png")
    plt.close()
    # plt.show()

def plot_plsc_figure_two(X_train_r,Y_train_r,X_test_r, Y_test_r, filename):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(X_train_r[:, 0], Y_train_r[:, 0], "ob", label="train")
    plt.plot(X_test_r[:, 0], Y_test_r[:, 0], "or", label="test")
    plt.xlabel("x scores")
    plt.ylabel("y scores")
    plt.title('Dimension 1: X vs Y (test corr = %.2f)' %
              np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1])
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc="best")

    plt.subplot(122)
    plt.plot(X_train_r[:, 1], Y_train_r[:, 1], "ob", label="train")
    plt.plot(X_test_r[:, 1], Y_test_r[:, 1], "or", label="test")
    plt.xlabel("x scores")
    plt.ylabel("y scores")
    plt.title('Comp. 2: X vs Y (test corr = %.2f)' %
              np.corrcoef(X_test_r[:, 1], Y_test_r[:, 1])[0, 1])
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc="best")

    plt.savefig(root_path + "/result/figure/cca/" + filename + "_two.png")
    plt.close()

def log_result(results, log_file, flag):
    np.save(log_file + '.npy', results)
    if flag == 2:
        np.savetxt(log_file + '.csv', results, fmt="%s", delimiter=",")


if __name__ == '__main__':

    feature_type = 'TD4'
    feature_action_sensitivity(feature_type)