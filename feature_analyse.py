# *-* coding: UTF-8 *-*
import os, sys
import numpy as np 
import data_load
import preprocess
import matplotlib.pyplot as plt
from preprocess import load_raw_dataset

root_path = os.getcwd()

def feature_distribute():
	# print root_path
	# sys.exit(0)
	
	actions = [i+1 for i in range(6)]
	actions = [1]
	for action in actions:
		print 'Generate action: ', action
		title = 'subject_1_feature_class_'+str(action)
	
		feature = np.load(root_path + '/train1_250_100/'+title+'.npy')
		channel_num = 34
		step = 4


		# feature_list = ['MAV', 'ZC', 'SSC', 'WL']
		feature_list = ['MAV']

		for start, feature_name in enumerate(feature_list):

			means_list = np.ones( (channel_num,) )
			stds_list = np.ones( (channel_num,) )
			# print type(means_list[0]), means_list.shape

			for i in xrange(channel_num):
				means_list[i] = np.mean( feature[:,start+i*step] ,axis=0)
				stds_list[i] = np.std( feature[:,start+i*step] ,axis=0)
			# print means_list, stds_list


			ind = np.arange(channel_num)
			width = 0.8

			# fig, ax = plt.subplots()
			plt.bar(ind, means_list, width, color='r', yerr=stds_list)

			plt.ylabel('Scores')
			plt.title(title)
			plt.xticks(ind+width)
			plt.xticklabels( ('1','2') )
			# plt.show()
			plt.savefig('result/figure/'+title+'_'+feature_name, dpi=120)
			plt.close()


	




if __name__ == '__main__':
	feature_distribute()


