# *-* coding: UTF-8 *-*
# !/usr/bin/python

import numpy as np
import math


def extract_MAV(data):
    ''' Mean Absolute Value '''
    res = 0
    for i in xrange(data.shape[0]):
        res += abs(data[i])
    res /= data.shape[0]
    return res


def extract_ZC(data):
    ''' Zeros Across '''
    res = 0
    for i in xrange(data.shape[0] - 1):
        if(data[i] * data[i + 1]) <= 0:
            res += 1
    return res


def extract_SSC(data):
    ''' Slope Sign Change '''
    res = 0
    for i in xrange(data.shape[0] - 2):
        if((data[i + 1] - data[i]) * (data[i + 1] - data[i + 2]) <= 0):
            res += 1
    return res


def extract_WL(data):
    ''' Waveform Length '''
    res = 0
    for i in xrange(data.shape[0] - 1):
        res += abs(data[i + 1] - data[i])
    return res


def extract_RMS(data):
    ''' Root Mean Square '''
    res = 0
    for i in xrange(data.shape[0]):
        res += data[i]**2
    res = math.sqrt(res / data.shape[0])
    return res


def extract_DRMS(data1, data2):
    res = 0
    res = extract_RMS(data1) - extract_RMS(data2)
    return res


def extract_TD4(data):
    res = np.zeros((4,))
    res[0] = extract_MAV(data)
    res[1] = extract_ZC(data)
    res[2] = extract_SSC(data)
    res[3] = extract_WL(data)
    return res


def extract_TD5(data):
    res = np.zeros((5,))
    res[0] = extract_MAV(data)
    res[1] = extract_ZC(data)
    res[2] = extract_SSC(data)
    res[3] = extract_WL(data)
    res[4] = extract_RMS(data)
    return res

if __name__ == '__main__':
    a = np.array([1.1, 2, 3, -4, -5, 6, -7, 8])
    b = np.array([1.5, 2.1, 3.8, -3.4, -4, 7.4, -7.7, 8.3])
    # print extract_MAV(a)
    # print extract_ZC(a)
    # print extract_SSC(a)
    # print extract_WL(a)
    # print extract_TD4(a)
    print extract_RMS(a), extract_RMS(b)
    print extract_DRMS(a, b)
