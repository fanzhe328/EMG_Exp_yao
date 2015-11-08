# *-* coding: UTF-8 *-*
import os
import numpy as np


root_path = os.getcwd()


def load_feature_dataset(dir='train1', subject='subject_1'):
    ''' 读取样本数据和类别数据， dir表示实验组名称，subject表示受试者，subject_all表示所有受试者'''
    if subject == 'subject_all':
        file_path = root_path + '/' + dir + '/subject_1'
        trains_1 = np.load(file_path + '_feature_trains.npy')
        classes_1 = np.load(file_path + '_feature_classes.npy')
        file_path = root_path + '/' + dir + '/subject_2'
        trains_2 = np.load(file_path + '_feature_trains.npy')
        classes_2 = np.load(file_path + '_feature_classes.npy')
        trains = np.concatenate((trains_1, trains_2), axis=0)
        classes = np.concatenate((classes_1, classes_2), axis=0)
        return trains, classes
    elif subject == 'subject_cross':
        file_path = root_path + '/' + dir + '/subject_1'
        trains_1 = np.load(file_path + '_feature_trains.npy')
        classes_1 = np.load(file_path + '_feature_classes.npy')
        file_path = root_path + '/' + dir + '/subject_2'
        trains_2 = np.load(file_path + '_feature_trains.npy')
        classes_2 = np.load(file_path + '_feature_classes.npy')
        return trains_1, classes_1, trains_2, classes_2
    else:
        file_path = root_path + '/' + dir + '/' + subject
        trains = np.load(file_path + '_feature_trains.npy')
        classes = np.load(file_path + '_feature_classes.npy')
        return trains, classes


def load_signal_dataset(dir='train1', subject='subject_1'):
    ''' 读取样本数据和类别数据， dir表示实验组名称，subject表示受试者，subject_all表示所有受试者'''

    if subject == 'subject_all':
        file_path = root_path + '/' + dir + '/subject_1'
        trains_1 = np.load(file_path + '_signal_trains.npy')
        classes_1 = np.load(file_path + '_signal_classes.npy')
        file_path = root_path + '/' + dir + '/subject_2'
        trains_2 = np.load(file_path + '_signal_trains.npy')
        classes_2 = np.load(file_path + '_signal_classes.npy')
        trains = np.concatenate((trains_1, trains_2), axis=0)
        classes = np.concatenate((classes_1, classes_2), axis=0)
        return trains, classes
    elif subject == 'subject_cross':
        file_path = root_path + '/' + dir + '/subject_1'
        trains_1 = np.load(file_path + '_signal_trains.npy')
        classes_1 = np.load(file_path + '_signal_classes.npy')
        file_path = root_path + '/' + dir + '/subject_2'
        trains_2 = np.load(file_path + '_signal_trains.npy')
        classes_2 = np.load(file_path + '_signal_classes.npy')
        return trains_1, classes_1, trains_2, classes_2
    else:
        file_path = root_path + '/' + dir + '/' + subject
        trains = np.load(file_path + '_signal_trains.npy')
        classes = np.load(file_path + '_signal_classes.npy')
        return trains, classes


def feature_normalized():
    ''' 特征归一化 2015-10-28 '''
    pass
