# *-* coding: UTF-8 *-*
import os, re
import numpy as np
from classifier import log_result

root_path = os.getcwd()


def intra_result_combination(
        fold_pre='result', win_inc='250_100', fold='TD4_data1_subject_1',
        file_pre='feat_TD4_', file_post='1'):
    res_fold = root_path+'/'+fold_pre+'/'+win_inc+'/'+fold
    dis = os.listdir(res_fold)
    os.chdir(res_fold)
    
    diss = []
    for di in dis:
    	if re.match('feat_TD4_(\w+)_1\.npy', di):
    		diss.append(di)

    # print res_fold
    ress = np.array([])
    ress_ylime = 0
    start_line = 0
    for di in diss:
    	data = np.load(di)
    	if start_line == 0:
    		ress = np.concatenate( (ress, data), axis=None)
    		start_line = 1
    	elif start_line == 1:
    		ress = np.concatenate( (ress, data[start_line:,:]), axis=None)
    	if ress_ylime == 0:
    		ress_ylime = data.shape[1]
    	print data.shape, ress.shape
    ress = ress.reshape( (-1, ress_ylime))
    # print ress.shape
    log_result(ress, 'feat_TD4_intra.npy', 2)

if __name__ == '__main__':

    intra_result_combination()
