# *-* coding: UTF-8 *-*
import os
import sys
import numpy as np
import data_load
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from preprocess import load_raw_dataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

root_path = os.getcwd()


def feature_stability():
    ''' 对每个特征，分析其在34个通道中的均值和方差 '''
    # print root_path
    # sys.exit(0)
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
            channel_num = 34
            step = 4

            feature_list = ['MAV', 'ZC', 'SSC', 'WL']
            # feature_list = ['MAV']

            for start, feature_name in enumerate(feature_list):

                means_list = np.ones((channel_num,))
                stds_list = np.ones((channel_num,))
                # print type(means_list[0]), means_list.shape

                for i in xrange(channel_num):
                    means_list[i] = np.mean(
                        feature[:, start + i * step], axis=0)
                    stds_list[i] = np.std(feature[:, start + i * step], axis=0)
                # print means_list, stds_list

                labels = np.arange(channel_num)
                ind = np.array([i * 2 for i in range(channel_num)])
                width = 0.8

                # plt.figure(num=1, figsize=(8,6))
                fig, ax = plt.subplots(figsize=(16, 6))
                ax.bar(ind, means_list, width, color='r', yerr=stds_list)

                ax.set_ylabel('Scores')
                # ax.set_xlim(0,64)
                ax.set_title(title + '_' + feature_name)
                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(labels + 1)
                # plt.show()
                plt.savefig('result/figure/stability/' +
                            title + '_' + feature_name, dpi=120)
                plt.close()


def feature_distribute(feature_num=1):
    exec('feature_distribute_' + str(feature_num) + '()')


def feature_distribute_1(channel_length=4):
    ''' 六个动作的单个特征的2D散点图分布，共34个通道  '''
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    subjects = ['subject_' + str(i + 1) for i in range(2)]
    # subjects = ['subject_1']
    for subject in subjects:
        title_pre = subject + '_feature_class_'

        channel_num = 34
        # channel_num = 2
        for channel in range(channel_num):

            feature_list = ['ZC', 'SSC', 'WL']
            # feature_list = ['MAV']
            for offset, feature_name in enumerate(feature_list):

                print 'Generate feature: ', feature_name

                # print channel
                # sys.exit(0)

                actions = [i + 1 for i in range(6)]
                # actions = [1]

                plt.figure(figsize=(8, 6))
                for action in actions:
                    filename = title_pre + str(action)
                    feature = np.load(
                        root_path + '/train1_250_100/' + filename + '.npy')
                    x = feature[:, channel * channel_length + offset]
                    y = np.ones(feature.shape[0]) * action

                    plt.scatter(x, y, c=colors[action], alpha=0.5)
                plt.ylabel('Scores')
                plt.title(subject + '-channel_' +
                          str(channel) + '-' + feature_name)
                # plt.show()
                # print channel*channel_length+offset, data.shape
                # sys.exit(0)
                plt.savefig('result/figure/distribute/' + subject +
                            '-channel_' + str(channel) + '-' + feature_name, dpi=120)
                plt.close()


def feature_distribute_2(channel_length=4):
    ''' 六个动作的两个特征组合的2D散点图分布，共34个通道  '''
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = ['o', '^', '+']
    # subjects = ['subject_'+str(i+1) for i in range(2)]
    subjects = ['subject_2']
    for subject in subjects:
        title_pre = subject + '_feature_class_'

        channel_num = 34
        # channel_num = 1
        for channel in range(channel_num):

            feature_list = ['MAV', 'ZC', 'SSC', 'WL']
            # feature_list = ['MAV']
            plt.figure(figsize=(8, 6))
            for offset1, feature_name1 in enumerate(feature_list):
                for offset2, feature_name2 in enumerate(feature_list):
                    if offset1 >= offset2:
                        continue

                    # print offset1,feature_name1, offset2, feature_name2
                    # sys.exit(0)
                    actions = [i + 1 for i in range(6)]
                    # actions = [1,2]
                    for action in actions:
                        filename = title_pre + str(action)
                        feature = np.load(
                            root_path + '/train1_250_100/' + filename + '.npy')
                        x = feature[:, channel * channel_length + offset1]
                        y = feature[:, channel * channel_length + offset2]

                        plt.scatter(x, y, c=colors[action], marker=markers[
                                       action % 3], alpha=0.5)

                    plt.ylabel(feature_name2)
                    plt.xlabel(feature_name1)
                    plt.title(subject + '-channel_' + str(channel) +
                              '-' + feature_name1 + '_' + feature_name2)
                    # plt.show()
                #   # print channel*channel_length+offset, data.shape
                #   # sys.exit(0)
                    plt.savefig('result/figure/distribute2/' + subject + '-channel_' +
                                str(channel) + '-' + feature_name1 + '_' + feature_name2, dpi=120)
                    plt.close()


def feature_distribute_3(channel_length=4):
    ''' 六个动作的三个特征组合的3D散点图分布，共34个通道  '''
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = ['o', '^', '+']

    # subjects = ['subject_'+str(i+1) for i in range(2)]
    subjects = ['subject_1']
    for subject in subjects:
        title_pre = subject + '_feature_class_'

        # channel_num = 34
        # channel_num = 18
        channel_num = 1

        for channel in range(channel_num):

            feature_list = ['MAV', 'ZC', 'SSC', 'WL']
            # feature_list = ['MAV']

            for offset1, feature_name1 in enumerate(feature_list):
                for offset2, feature_name2 in enumerate(feature_list):
                    for offset3, feature_name3 in enumerate(feature_list):
                        if offset1 >= offset2 or offset2 >= offset3:
                            continue

                        actions = [i + 1 for i in range(6)]
                        actions = [1, 2]

                        fig = plt.figure(figsize=(8, 6))
                        # ax = fig.add_subplot(111, projection='3d')
                        ax = Axes3D(fig)
                        for action in actions:
                            filename = title_pre + str(action)
                            feature = np.load(
                                root_path + '/train1_250_100/' + filename + '.npy')
                            xs = feature[:, channel * channel_length + offset1]
                            ys = feature[:, channel * channel_length + offset2]
                            zs = feature[:, channel * channel_length + offset3]

                            ax.scatter(xs, ys, zs, c=colors[action], marker=markers[
                                       action % 3], alpha=0.5)

                        ax.set_xlabel(feature_name1)
                        ax.set_ylabel(feature_name2)
                        ax.set_zlabel(feature_name3)
                        ax.legend(actions)
                        plt.title(subject + '-channel_' + str(channel) + '-' +
                                  feature_name1 + '_' + feature_name2 + '_' + feature_name3)
                        plt.show()

                        # plt.savefig(
                        #     'result/figure/distribute3/' + subject + '-channel_'
                        #     + str(channel) + '-' + feature_name1 + '_' +
                        #     feature_name2 + '_' + feature_name3,
                        #     dpi=120)
                        # plt.close()


def feature_distribute_3_projection(channel_length=4, projection='pca'):
    ''' 六个动作的三个特征组合的2D映射分布，共34个通道  '''
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = ['o', '+', 'v', '^', '*', 'x']
    sample_len = 150
    subjects = ['subject_'+str(i+1) for i in range(2)]          # 受试者
    # subjects = ['subject_1']
    for subject in subjects:
        title_pre = subject + '_feature_class_'

        channel_num = 34                                        # 通道
        # channel_num = 18
        # channel_num = 1
        for channel in range(channel_num):

            feature_list = ['MAV', 'ZC', 'SSC', 'WL']
            # feature_list = ['MAV']

            for offset1, feature_name1 in enumerate(feature_list):
                for offset2, feature_name2 in enumerate(feature_list):
                    for offset3, feature_name3 in enumerate(feature_list):
                        if offset1 >= offset2 or offset2 >= offset3:
                            continue
                        actions = [i + 1 for i in range(6)]         # 动作
                        # actions = [1, 2]
                        fig = plt.figure(figsize=(8, 6))
                        ax = fig.add_subplot()
                        trains = np.array([])
                        targets = np.array([], np.int)
                        for action in actions:
                            filename = title_pre + str(action)
                            feature = np.load(
                                root_path + '/train1_250_100/' + filename + '.npy')
                            xs = feature[:sample_len, channel * channel_length + offset1]
                            ys = feature[:sample_len, channel * channel_length + offset2]
                            zs = feature[:sample_len, channel * channel_length + offset3]
                            # if offset2 == 2 and offset3 == 3:
                            #     print ys.mean(), ys.std(), zs.mean(), zs.std()
                            #     sys.exit(0)
                            train = np.concatenate(
                                [xs.reshape((-1, 1)), ys.reshape((-1, 1)), ys.reshape((-1, 1))], axis=1)
                            target = np.ones(train.shape[0], np.int) * action
                            # print train.shape, target.shape, target[0, 0:5]
                            trains = np.concatenate((trains, train), axis=None)
                            targets = np.concatenate(
                                (targets, target), axis=None)

                            # sys.exit(0)
                        trains = trains.reshape((-1, 3))
                        
                        # print trains.shape, targets.shape
                        if projection == 'pca':
                            pca = PCA(n_components=2)
                            X_r = pca.fit(trains).transform(trains)
                        elif projection == 'lda':
                            lda = LinearDiscriminantAnalysis(n_components=2)
                            X_r = lda.fit(trains, targets).transform(trains)

                        # print X_r.shape, targets.shape, X_r, targets
                        # print X_r[targets == 0, 0]
                        # print X_r[0:100, 0]
                        for action in actions:
                            plt.scatter(X_r[targets == action, 0], X_r[targets == action, 1], 
                                c=colors[action], marker=markers[
                                action % 1], alpha=0.5, label=action)
                        plt.legend()
                        plt.title(subject + '-channel_' + str(channel) + '-' + projection + '-'
                                  +feature_name1 + '_' + feature_name2 + '_' + feature_name3)
                        
                        # plt.show()
                        plt.savefig(
                            'result/figure/distribute3_proj/' + subject + '-channel_'
                            + str(channel) + '-' + projection + '-' + feature_name1 + '_' +
                            feature_name2 + '_' + feature_name3,
                            dpi=120)
                        plt.close()

def feature_distribute_4_projection(channel_length=4, projection='pca'):
    ''' 六个动作的四个特征组合的2D映射分布，共34个通道  '''
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = ['o', '+', 'v', '^', '*', 'x']
    sample_len = 100
    subjects = ['subject_'+str(i+1) for i in range(2)]          # 受试者
    # subjects = ['subject_1']
    for subject in subjects:
        title_pre = subject + '_feature_class_'

        channel_num = 34                                        # 通道
        # channel_num = 18
        # channel_num = 1
        for channel in range(channel_num):

            feature_list = ['MAV', 'ZC', 'SSC', 'WL']
            # feature_list = ['MAV']

            actions = [i + 1 for i in range(6)]                 # 动作
            # actions = [1, 2]
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
            trains = np.array([])
            targets = np.array([], np.int)
            for action in actions:
                filename = title_pre + str(action)
                feature = np.load(
                    root_path + '/train1_250_100/' + filename + '.npy')
                train = feature[:sample_len, channel * channel_length : channel * channel_length+4]
                target = np.ones(train.shape[0], np.int) * action
                # print train.shape, target.shape, target[0, 0:5]
                trains = np.concatenate((trains, train), axis=None)
                targets = np.concatenate(
                    (targets, target), axis=None)

                # sys.exit(0)
            trains = trains.reshape((-1, 4))
            
            # print trains.shape, targets.shape
            if projection == 'pca':
                pca = PCA(n_components=2)
                X_r = pca.fit(trains).transform(trains)
            elif projection == 'lda':
                lda = LinearDiscriminantAnalysis(n_components=2)
                X_r = lda.fit(trains, targets).transform(trains)

            for action in actions:
                plt.scatter(X_r[targets == action, 0], X_r[targets == action, 1], 
                    c=colors[action], marker=markers[
                    action % 1], alpha=0.5, label=action)
            plt.legend()
            plt.title(subject + '-channel_' + str(channel) + '-' + projection + '-TD4')
            
            # plt.show()
            plt.savefig(
                'result/figure/distribute4_proj/' + subject + '-channel_'
                + str(channel) + '-' + projection + '-TD4',
                dpi=120)
            plt.close()

if __name__ == '__main__':
    # feature_stability()
    # # print 'test'
    # feature_distribute(3)
    feature_distribute_4_projection(4, 'pca')
    feature_distribute_4_projection(4, 'lda')
    # feature_distribute_3_projection(4, 'pca')
    # feature_distribute_3_projection(4, 'lda')

