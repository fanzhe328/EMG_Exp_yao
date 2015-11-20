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

def data_trainsform(data):
    idx_S0 = np.array([17, 19, 45, 47]) - 1
    chan_num = len(idx_S0)
    res = np.zeros( (data.shape[0], 36) )       # 4*9=36, 9表示电极位置的九种情况：原点，上1cm，上2cm，下1cm，下2cm...
    
    i = 0
    res[:,i:i+chan_num] = data[:,idx_S0]        # S0
    
    i += chan_num
    res[:,i:i+chan_num] = data[:,idx_S0-8]                  # 上1cm
        
    i += chan_num
    res[:,i:i+chan_num] = data[:,idx_S0-[13,13,15,15]]      # 上2cm
    
    i += chan_num
    res[:,i:i+chan_num] = data[:,idx_S0+7]                  # 下1cm
    
    i += chan_num
    res[:,i:i+chan_num] = data[:,idx_S0+13]                 # 下2cm
    
    i += chan_num
    res[:,i:i+chan_num] = data[:,idx_S0-1]                  # 左1cm
    
    i += chan_num
    res[:,i:i+chan_num] = data[:,idx_S0-2]                  # 左2cm
    
    i += chan_num
    res[:,i:i+chan_num] = data[:,idx_S0+1]                  # 右1cm
    
    i += chan_num
    res[:,i:i+chan_num] = data[:,idx_S0+2]                  # 右2cm

    return res

def load_raw_dataset(dir='data1', subject='subject_1'):
    ''' 姚传存师兄的数据集'''
    dir_train = root_path + '/' + dir
    # 原始文件名称，已经截取动作片， dir表示实验组名称，subject表示受试者
    mat_file = dir_train + '/' + subject + '.mat'

    data = sio.loadmat(mat_file)
    classes = data['motions']
    class_idx = np.array([2,7,3,11,10,5,8]) - 1         # 选取七个动作，FPG，KG，FP，WF，WE，HC，NM
    times = data['times']
    chan_num = data['channels']
    trainingdata = data['data']
    trains = []
    # print trainingdata[0][0]['emg'].shape, len(trainingdata[0][0])
    # sys.exit(0)
    
    for i in range(len(trainingdata)):
        for j in range(len(trainingdata[i])):
            # len(trainingdata[j]) 表示训练动作的个数，data4中有11个动作
            trains.append(data_trainsform(trainingdata[i][j]['emg']))
            # trains.append(trainingdata[i][j]['emg'])
    trains = np.array(trains[:])[class_idx,:,:]
    # print trains.shape
    # sys.exit(0)
    # trains = trains[class_idx,:,:]
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


def generate_feature_dataset(dir='train1', subject='subject_1', feature_type='TD4'):
    ''' 读取特征数据集，并生成特征数组和类别数组 '''
    print "----generate_feature_dataset, ", dir, subject
    dir_train = root_path + '/' + dir
    dis = os.listdir(dir_train)
    os.chdir(dir_train)

    # x = dis[3]
    # print x
    # print re.match(r'feature_class_(\d+)\.npy', x).group(1)
    match_template = subject + '_feat_' + feature_type + '_action_(\d+)\.npy'
    diss = []
    for fi in dis:
        if re.match(match_template, fi):
            diss.append(fi)
    
    diss.sort(key=lambda x: int(
        re.match(match_template, x).group(1)))
    # dis.sort(key = lambda x:int(re.match('feature_class_(\d+)\.npy', x).group(1)))
    trains = np.array([])
    targets = np.array([], np.int)
    trains_ylim = 0
    # print diss
    # sys.exit(0)
    for fi in diss:
        data = np.load(fi)
        target_num = int(
            re.match(match_template, fi).group(1))
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
    np.save(file_path + '_feat_'+feature_type +'_trains.npy', trains)
    np.save(file_path + '_feat_'+feature_type +'_classes.npy', targets)
    np.savetxt(file_path + '_feat_'+feature_type +'_trains.csv',
               trains, fmt="%s", delimiter=",")  # 保存数据
    np.savetxt(file_path + '_feat_'+feature_type +'_classes.csv',
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


def feature_extract(raw_data, target, window, overlap, sample_rate, feature_type='TD4', out_dir='train1', subject='subject_1', feat_num=4):
    ''' 生成数据样本集合（提取特征），参数：原始数据，类别，时间窗（250），重叠窗（100），采样率（1024）'''
    # print raw_data.shape, target, window, overlap
    print "----feature_extract target ", target, "...................."
    start_time = time.time()
    winsize = (int)(sample_rate * window * 0.001)
    incsize = (int)(sample_rate * overlap * 0.001)
    start = 0
    # print winsize, incsize

    x_dim = (raw_data.shape[0] - winsize) / incsize + 1
    y_dim = raw_data.shape[1] * feat_num

    index = 0
    trains = np.zeros((x_dim, y_dim))
    while start + winsize < raw_data.shape[0]:
        train = np.array([])
        for i in range(raw_data.shape[1]):
            if feature_type == 'TD4':
                cur_win_feature = fe.extract_TD4(
                    raw_data[start:start + winsize, i])		# 提取TD4（MAV，ZC，SSC，WL）四种时域特征
            if feature_type == 'TD5':
                cur_win_feature = fe.extract_TD5(
                    raw_data[start:start + winsize, i])     # 提取TD4（MAV，ZC，SSC，WL）四种时域特征
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
        subject + '_feat_' + feature_type + '_action_' + str(target)

    if(not new_fold(root_path + '/' + out_dir)):
        sys.exit(1)

    np.save(log_file + '.npy', trains)
    np.savetxt(log_file + '.csv', trains, fmt="%s", delimiter=",")
    print "----feature_extract target ", target, " over, time elapsed:", time.time() - start_time


def data_preprocess(input_dir='data1', train_dir='train1', feature_type='TD4', 
                        subject_list=[1], winsize=250, incsize=100, samrate=1024, feat_num=4):
    ''' 预处理，生成未标准化（Z-Score）的数据样本和类别 '''
    print "data_preprocess................."
    start_time = time.time()

    for sub in subject_list:
        print "----Running ", sub, '....................'
        trains = load_raw_dataset(input_dir, sub)

        for i in range(len(trains)):                # 动作的数量
            feature_extract(trains[i], i + 1, winsize, incsize,     #提取第i个动作的特征
                            samrate, feature_type, train_dir, sub, feat_num)
            ### generate_samples(trains[i], i+1, winsize, incsize, samrate)
        
        generate_feature_dataset(train_dir, sub, feature_type)
        # sys.exit(0)
        # generate_signal_dataset(train_dir, sub)
    print "data_preprocess time elapsed: ", time.time() - start_time


def data_normalize(trains):
    trains_scale = preprocessing.scale(trains)
    return trains_scale


if __name__ == '__main__':
    input_dir = 'data4'
    train_dir = 'train4'
    
    feature_type = 'TD4'
    feat_num = 4
    
    feature_type = 'TD5'
    feat_num = 5
    
    subject_list = ['subject_' + str(i) for i in range(1, 6)]
    # print subject_list
    # sys.exit(0)
    winsize = 250
    incsize = 100
    samrate=1024
    train_dir = train_dir+'_'+str(winsize)+'_'+str(incsize)
    data_preprocess(input_dir, train_dir, feature_type, subject_list,
                    winsize, incsize, samrate, feat_num)
    print 'TD4 finished'
    # feature_type = 'TD5'
    # data_preprocess(input_dir, train_dir, feature_type, subject_list,
    #                 winsize, incsize, samrate)
