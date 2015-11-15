# *-* coding: UTF-8 *-*
import os
import sys
import numpy as np


root_path = os.getcwd()


def load_feature_dataset(dir='train4_250_100', subject='subject_1'):
    ''' 读取样本数据和类别数据， dir表示实验组名称，subject表示受试者，subject_all表示所有受试者'''
    file_path = root_path + '/' + dir + '/' + subject
    trains = np.load(file_path + '_feature_trains.npy')
    classes = np.load(file_path + '_feature_classes.npy')
    return trains, classes


def load_signal_dataset(dir='train1', subject='subject_1'):
    ''' 读取样本数据和类别数据， dir表示实验组名称，subject表示受试者，subject_all表示所有受试者'''
    file_path = root_path + '/' + dir + '/' + subject
    trains = np.load(file_path + '_signal_trains.npy')
    classes = np.load(file_path + '_signal_classes.npy')
    return trains, classes


def feature_normalized():
    ''' 特征归一化 2015-10-28 '''
    pass


if __name__ == '__main__':
    trains, targets = load_feature_dataset()
    print trains.shape, targets.shape

