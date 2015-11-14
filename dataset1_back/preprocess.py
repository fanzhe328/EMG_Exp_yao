# *-* coding: UTF-8 *-*
import os
import sys
import time
import re
import scipy.io as sio
import numpy as np
import feature_extraction as fe
from file_script import new_fold
from sklearn import preprocessing

root_path = os.getcwd()


def load_raw_dataset(dir='data1', subject='subject_1'):
    ''' 姚传存师兄的数据集'''
    dir_train = root_path + '/' + dir
    # 原始文件名称，已经截取动作片， dir表示实验组名称，subject表示受试者
    mat_file = dir_train + '/' + subject + '.mat'

    data = sio.loadmat(mat_file)
    if dir == 'data1':
        trainingdata = data['trainingdata']		# 提取mat文件中的原始信号数据
    elif dir == 'data4':
        trainingdata = data['data']
    trains = []

    for i in range(len(trainingdata)):
        # len(trainingdata[i]) 表示训练动作的个数， data1中有15个动作， data4中有11个动作
        for j in range(len(trainingdata[i])):
            if dir == 'data1':
                trains.append(trainingdata[i][j]['trainingdata'])
            elif dir == 'data4':
                trains.append(trainingdata[i][j]['emg'])
    return trains


def generate_signal_dataset(dir='train1', subject='subject_1'):
    ''' 读取信号数据集，并生成信号数组和类别数组 '''
    print "----generate_signal_dataset, ", dir, subject
    dir_train = root_path + '/' + dir
    dis = os.listdir(dir_train)
    os.chdir(dir_train)

    diss = []
    for fi in dis:
        if re.match(subject + '_signal_class_(\d+)\.npy', fi):
            diss.append(fi)
    diss.sort(key=lambda x: int(
        re.match(subject + '_signal_class_(\d+)\.npy', x).group(1)))

    trains = np.array([])
    targets = np.array([], np.int)
    trains_ylim = 0
    for fi in diss:
        data = np.load(fi)
        print 'fi:', fi, ' :', data.shape
        target_num = int(
            re.match(subject + '_signal_class_(\d+)\.npy', fi).group(1))
        if trains_ylim == 0:
            trains_ylim = data.shape[1]
        target_temp = np.ones((data.shape[0],), np.int)
        np.multiply(target_temp, target_num, target_temp)
        trains = np.concatenate((trains, data), axis=None)
        targets = np.concatenate((targets, target_temp), axis=None)
    print type(targets)
    trains = trains.reshape((-1, trains_ylim))
    file_path = root_path + '/' + dir + '/' + subject
    np.save(file_path + '_signal_trains.npy', trains)
    np.save(file_path + '_signal_classes.npy', targets)
    np.savetxt(file_path + '_signal_trains.csv',
               trains, fmt="%s", delimiter=",")  # 保存数据
    np.savetxt(file_path + '_signal_classes.csv',
               targets, fmt="%s", delimiter=",")  # 保存数据


def generate_feature_dataset(dir='train1', subject='subject_1'):
    ''' 读取特征数据集，并生成特征数组和类别数组 '''
    print "----generate_feature_dataset, ", dir, subject
    dir_train = root_path + '/' + dir
    dis = os.listdir(dir_train)
    os.chdir(dir_train)

    # x = dis[3]
    # print x
    # print re.match(r'feature_class_(\d+)\.npy', x).group(1)
    diss = []
    for fi in dis:
        if re.match(subject + '_feature_class_(\d+)\.npy', fi):
            diss.append(fi)
    diss.sort(key=lambda x: int(
        re.match(subject + '_feature_class_(\d+)\.npy', x).group(1)))
    # dis.sort(key = lambda x:int(re.match('feature_class_(\d+)\.npy', x).group(1)))
    trains = np.array([])
    targets = np.array([], np.int)
    trains_ylim = 0
    for fi in diss:
        data = np.load(fi)
        target_num = int(
            re.match(subject + '_feature_class_(\d+)\.npy', fi).group(1))
        # target_num = (int)target_num
        # print type(target_num)
        if trains_ylim == 0:
            trains_ylim = data.shape[1]
        target_temp = np.ones((data.shape[0],), np.int)
        np.multiply(target_temp, target_num, target_temp)
        trains = np.concatenate((trains, data), axis=None)
        targets = np.concatenate((targets, target_temp), axis=None)
    trains = trains.reshape((-1, trains_ylim))
    file_path = root_path + '/' + dir + '/' + subject
    np.save(file_path + '_feature_trains.npy', trains)
    np.save(file_path + '_feature_classes.npy', targets)
    np.savetxt(file_path + '_feature_trains.csv',
               trains, fmt="%s", delimiter=",")  # 保存数据
    np.savetxt(file_path + '_feature_classes.csv',
               targets, fmt="%s", delimiter=",")  # 保存数据
    print '----Save success, dir', dir, ', subject:', subject
    # print data.shape, target_temp.shape, target_num, target_temp[0]
    # for fi in dis:
    # 	data = np.load(fi)
    # 	trains.append(data)


def generate_samples(raw_data, target, window, overlap, sample_rate, subject="subject_1", out_dir='train1'):
    ''' 生成数据样本集合（不提取特征），参数：原始数据，类别，时间窗（250），重叠窗（100），采样率（1024）'''
    # 原始数据，每一行为一个时间点下所有通道的值，每一列为一个通道在一段时间的信号量变化
    # print raw_data.shape, target, window, overlap
    print "generate_samples target ", target, " ...................."
    start_time = time.time()
    winsize = (int)(sample_rate * window * 0.001) 		# 每个时间窗中含有的数据点的数量
    incsize = (int)(sample_rate * overlap * 0.001) 		# 每个重叠时间窗中含有的数据点的数量
    start = 0
    # print winsize, incsize

    x_dim = (raw_data.shape[0] - winsize) / incsize + 1		# 矩阵的行数
    y_dim = raw_data.shape[1] * winsize					# 矩阵的列数

    index = 0
    trains = np.zeros((x_dim, y_dim))					# 最终生成的训练数据（信号层次）
    # print trains.shape, trains.dtype
    while start + winsize < raw_data.shape[0]:
        train = np.array([])
        for i in range(raw_data.shape[1]):				# 读取每个通道的数据
            cur_win_signal = raw_data[start:start + winsize, i]
            train = np.concatenate(
                (train, cur_win_signal), axis=None)  # 将一个时间窗内的所有通道的数据组合
        # print train.shape
        trains[index] = train
        index += 1
        start += incsize

    log_file = root_path + '/' + out_dir + '/' + \
        subject + '_signal_class_' + str(target)
    np.save(log_file + '.npy', trains)								# 保存数据
    np.savetxt(log_file + '.csv', trains, fmt="%s", delimiter=",")  # 保存数据
    print "generate_samples target ", target, " over, time elapsed:", time.time() - start_time


def feature_extract(raw_data, target, window, overlap, sample_rate, feature_type='TD4', out_dir='train1', subject='subject_1'):
    ''' 生成数据样本集合（提取特征），参数：原始数据，类别，时间窗（250），重叠窗（100），采样率（1024）'''
    # print raw_data.shape, target, window, overlap
    print "----feature_extract target ", target, "...................."
    start_time = time.time()
    winsize = (int)(sample_rate * window * 0.001)
    incsize = (int)(sample_rate * overlap * 0.001)
    start = 0
    # print winsize, incsize

    x_dim = (raw_data.shape[0] - winsize) / incsize + 1
    y_dim = raw_data.shape[1] * 4

    index = 0
    trains = np.zeros((x_dim, y_dim))
    while start + winsize < raw_data.shape[0]:
        train = np.array([])
        for i in range(raw_data.shape[1]):
            if feature_type == 'TD4':
                cur_win_feature = fe.extract_TD4(
                    raw_data[start:start + winsize, i])		# 提取TD4（MAV，ZC，SSC，WL）四种时域特征
            # elif feature_type:
            # cur_win_feature = fe.extract_TD4_AR(raw_data[start:start+winsize,
            # i])           # 提取TD4（MAV，ZC，SSC，WL）四种时域特征+AR特征
            train = np.concatenate(
                (train, cur_win_feature), axis=None)  # 对每个时间窗提取特征
        # print train.shape
        trains[index] = train
        index += 1
        start += incsize

    log_file = root_path + '/' + out_dir + '/' + \
        subject + '_feature_class_' + str(target)

    if(not new_fold(root_path + '/' + out_dir)):
        sys.exit(1)

    np.save(log_file + '.npy', trains)
    np.savetxt(log_file + '.csv', trains, fmt="%s", delimiter=",")
    print "----feature_extract target ", target, " over, time elapsed:", time.time() - start_time


def data_preprocess(input_dir='data1', train_dir='train1', feature_type='TD4', subject_list=[1], winsize=250, incsize=100, samrate=1024):
    ''' 预处理，生成未标准化（Z-Score）的数据样本和类别 '''
    print "data_preprocess................."
    start_time = time.time()

    for sub in subject_list:
        print "----Running ", sub, '....................'
        trains = load_raw_dataset(input_dir, sub)
        for i in range(len(trains)):                # 动作的数量
            feature_extract(trains[i], i + 1, winsize, incsize,     #提取第i个动作的特征
                            samrate, feature_type, train_dir, sub)
            # generate_samples(trains[i], i+1, winsize, incsize, samrate)
        generate_feature_dataset(train_dir, sub)
        # generate_signal_dataset(train_dir, sub)
    print "data_preprocess time elapsed: ", time.time() - start_time


def data_normalize(trains):
    trains_scale = preprocessing.scale(trains)
    return trains_scale


if __name__ == '__main__':
    input_dir = 'data1'
    train_dir = 'train1'
    feature_type = 'TD4'
    subject_list = ['subject_' + str(i) for i in range(1, 3)]
    data_preprocess(input_dir, train_dir, feature_type, subject_list)
