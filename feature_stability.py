# *-* coding: UTF-8 *-*
import os
import sys
import numpy as np
import data_load
import matplotlib.pyplot as plt

root_path = os.getcwd()

from preprocess import load_raw_dataset



def feature_stability_O2O():
    ''' 对每个特征，分析其在不同移位程度的的均值和方差 '''
    # print root_path
    # sys.exit(0)
    channel_pos_list = ['O',                                # 中心位置
                    'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2',
                    'E1', 'E2', 'F1', 'F2', 'G1', 'G2', 'H1', 'H2']  #
    subjects = ['subject_' + str(i + 1) for i in range(2)]
    # print subjects
    # sys.exit(0)
    for subject in subjects:
        title_pre = subject + '_feature_class_'
        actions = [i + 1 for i in range(6)]
        # actions = [1]
        for action in actions:
            print 'Generate action: ', action
            title = title_pre + str(action)

            feature = np.load(root_path + '/train1_250_100/' + title + '.npy')
            channel_num = 34                # 实验中的通道总数
            channel_group_len = 2           # 双通道
            channel_span = 17               # 双通道中一组电极两个位置的跨度
            feat_num = 4

            feature_list = ['MAV', 'ZC', 'SSC', 'WL']
            # feature_list = ['MAV']

            for feat_ind, feature_name in enumerate(feature_list):

                means_list = np.ones((channel_num/channel_group_len,))
                stds_list = np.ones((channel_num/channel_group_len,))
                # print type(means_list[0]), means_list.shape
                # print feature.shape
                for i in xrange(channel_num/channel_group_len):
                    # print i*feat_num+feat_ind, i*feat_num+feat_ind+channel_span*feat_num

                    temp = np.concatenate(
                        (feature[:, i*feat_num+feat_ind], feature[:, i*feat_num+feat_ind+channel_span*feat_num]),
                        axis = None)
                    # print temp.shape
                    # sys.exit(0)
                    # temp = feature[:, feat_ind + i * step]
                    means_list[i] = np.mean(temp, axis=0)
                    stds_list[i] = np.std(temp, axis=0)

                # print means_list, stds_list
            # sys.exit(0)

                labels = np.arange(channel_num)
                ind = np.array([i * 2 for i in range(channel_num/channel_group_len)])
                width = 0.8

                # plt.figure(num=1, figsize=(8,6))
                fig, ax = plt.subplots(figsize=(16, 6))
                ax.bar(ind, means_list, width, color='r', yerr=stds_list)

                ax.set_ylabel('Scores')
                # ax.set_xlim(0,64)
                ax.set_title(title + '_O2O_' + feature_name)
                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(channel_pos_list)
                # plt.show()
                # if feat_ind == 1:
                #     sys.exit(0)
                plt.savefig('result/figure/stability/' +
                            title + '_O2O_' + feature_name, dpi=120)
                plt.close()

def feature_stability_O2A():
    ''' 对每个特征，分析其在不移位和在所谓情况下的均值和方差 '''
    # print root_path
    # sys.exit(0)
    channel_pos_list = ['O',                                # 中心位置
                    'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2',
                    'E1', 'E2', 'F1', 'F2', 'G1', 'G2', 'H1', 'H2']  #
    subjects = ['subject_' + str(i + 1) for i in range(2)]
    feature_list = ['MAV', 'ZC', 'SSC', 'WL']
    xtickslabels_post = ['O', 'A']
    xtickslabels = [x+'_'+y for x in feature_list for y in xtickslabels_post]
    # print xtickslabels
    # print subjects
    # sys.exit(0)
    for subject in subjects:
        title_pre = subject + '_feature_class_'
        actions = [i + 1 for i in range(6)]
        # actions = [1]
        for action in actions:
            print 'Generate action: ', action
            title = title_pre + str(action)

            feature = np.load(root_path + '/train1_250_100/' + title + '.npy')
            channel_num = 34                # 实验中的通道总数
            channel_group_len = 2           # 双通道
            channel_span = 17               # 双通道中一组电极两个位置的跨度
            feat_num = 4

            
            # feature_list = ['MAV']
            data_len = 2*len(feature_list)
            means_list = np.ones((data_len,))
            stds_list = np.ones((data_len,))
            for feat_ind, feature_name in enumerate(feature_list):

                # print type(means_list[0]), means_list.shape
                # print feature.shape
                feature_O = np.concatenate(
                        (feature[:, feat_ind], feature[:, feat_ind+channel_span*feat_num]),
                        axis = None)
                feature_A = np.array([])
                for i in xrange(channel_num/channel_group_len):
                    # print i*feat_num+feat_ind, i*feat_num+feat_ind+channel_span*feat_num

                    temp = np.concatenate(
                        (feature[:, i*feat_num+feat_ind], feature[:, i*feat_num+feat_ind+channel_span*feat_num]),
                        axis = None)
                    feature_A = np.concatenate( (feature_A, temp), axis=None )
                    # print feature_A.shape
                    # sys.exit(0)
                    # temp = feature[:, feat_ind + i * step]
                
                means_list[feat_ind*2] = np.mean(feature_O, axis=0)
                stds_list[feat_ind*2] = np.std(feature_O, axis=0)
                means_list[feat_ind*2+1] = np.mean(feature_A, axis=0)
                stds_list[feat_ind*2+1] = np.std(feature_A, axis=0)

                # print means_list, stds_list
                # sys.exit(0)
                bar_num = 2
                labels = np.arange(bar_num)
                ind = np.array([i * 1 for i in range(bar_num)])
                width = 0.3

                # plt.figure(num=1, figsize=(8,6))
                fig, ax = plt.subplots(figsize=(8,6))
                ax.bar(ind, means_list[feat_ind*bar_num:feat_ind*bar_num+bar_num], width, 
                    color='r', yerr=stds_list[feat_ind*bar_num:feat_ind*bar_num+bar_num])

                ax.set_ylabel('Scores')
                # ax.set_xlim(0,64)
                ax.set_title(title + '_O2A_' + feature_name)
                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(xtickslabels_post)
                # plt.show()
                # if action == 1:
                #     sys.exit(0)
                plt.savefig('result/figure/stability/' +
                            title + '_O2A_' + feature_name, dpi=120)
                plt.close()
            
            bar_num = data_len
            labels = np.arange(bar_num)
            ind = np.array([i * 1 for i in range(bar_num)])
            width = 0.3

            # plt.figure(num=1, figsize=(8,6))
            fig, ax = plt.subplots(figsize=(8,6))
            ax.bar(ind, means_list, width, color='r', yerr=stds_list)

            ax.set_ylabel('Scores')
            # ax.set_xlim(0,64)
            ax.set_title(title + '_O2A')
            ax.set_xticks(ind + width / 2)
            ax.set_xticklabels(xtickslabels)
            # plt.show()
            plt.savefig('result/figure/stability/' +
                        title + '_O2A', dpi=120)
            plt.close()
            # if action == 1:
            #     sys.exit(0)

if __name__ == '__main__':
    # feature_stability_O2O()
    feature_stability_O2A()

