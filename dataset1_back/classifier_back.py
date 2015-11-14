
import numpy as np
import os, time, sys
import random
import sklearn
import sklearn.lda, sklearn.qda
import sklearn.naive_bayes
import sklearn.cross_validation
import sklearn.svm

from noise_simulation import proportion_simu
# from sklearn import linear_model
# from sklearn.lda import LDA
# from sklearn.qda import QDA
# from sklearn.svm import SVC, LinearSVC
# from sklearn.naive_bayes import GaussianNB

root_path = os.getcwd()
feat_map=['rms','ar','iav','mav','mav1','mav2','zc','ssc','wl','aac','dasdv','var','tm3','tm4','tm5','vorder','log','mnf','mdf','pkf','mnp','ttp','vcf','sm1','sm2','sm3','ssi',]
import_module = ("numpy as np","sklearn.cross_validation","sklearn.lda","sklearn.qda","sklearn.naive_bayes", "sklearn.svm")

def log_result(results, log_file, flag):
	np.save(log_file + '.npy', results)
	if flag == 2:
		np.savetxt(log_file+'.csv', results, fmt="%s", delimiter=",")

def new_fold(log_fold):
	if os.path.isdir(log_fold) == False:
		try:
			os.makedirs(log_fold)
		except:
			print "Can not create log fold! "
			return False
	return True

def training_lda_single_feat(my_clfs, trains, classes, **kw):
	print "Training lda single feat............."
	start_time = time.time()
	cv = 5
	if(kw.has_key('log_fold')):
		log_fold = root_path + '/result/' + kw['log_fold']
	if os.path.isdir(log_fold) == False:
		try:
			os.makedirs(log_fold)
		except:
			print "Can not create log fold! "
			return
	os.chdir(log_fold)

	results = []
	results.append(['Feat','Algorithm', 'Accuracy', 'std'])
	log_file = 'feat_1_cv_'+str(cv)

	jobs = []
	
	clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None, n_components=4, store_covariance=False, tol=0.0001)
	for i in range(len(feat_map)):
		# if  i == 12 or i == 13 or i == 14 or i == 15:
		# 	continue
		if i != 3 and i != 6 and i != 7 and i != 8:
			continue
		scores = sklearn.cross_validation.cross_val_score(clf, trains[i], classes, cv=cv)
		# print "score:", scores.mean()," type of scores: ", type(scores)
		results.append([feat_map[i], "LDA(solver='svd', shrinkage=None, priors=None, n_components=4, store_covariance=False, tol=0.000, cv=5)", scores.mean(), scores.std()])

	clf = sklearn.lda.LDA(solver='lsqr', shrinkage='auto', priors=None, n_components=4, store_covariance=False, tol=0.0001)
	for i in range(len(feat_map)):
		if i != 3 and i != 6 and i != 7 and i != 8:
			continue
		scores = sklearn.cross_validation.cross_val_score(clf, trains[i], classes, cv=cv)
		results.append([feat_map[i], "LDA(solver='lsqr', shrinkage=auto, priors=None, n_components=3, store_covariance=False, tol=0.000, cv=5)", scores.mean(), scores.std()])

	# Error: numpy.linalg.linalg.LinAlgError: the leading minor of order 3 of 'b' is not positive definite. The factorization of 'b' could not be completed and no eigenvalues or eigenvectors were computed.
	# clf = sklearn.lda.LDA(solver='eigen', shrinkage='auto', priors=None, n_components=4, store_covariance=False, tol=0.0001)
	# for i in range(len(feat_map)):
	# 	if i != 3 and i != 6 and i != 7 and i != 8:
	# 		continue
	# 	scores = sklearn.cross_validation.cross_val_score(clf, trains[i], classes, cv=cv)
	# 	results.append([feat_map[i], "LDA(solver='eigen', shrinkage=auto, priors=None, n_components=3, store_covariance=False, tol=0.000, cv=5)", scores.mean(), scores.std()])

	log_result(results, log_file, 2)
	print "Training lda single feat time elapsed: ", time.time()-start_time

def training_lda_TD4(my_clfs, trains, classes, **kw):
	start_time = time.time()
	print "training TD4............."
	cv = 5
	
	if(kw.has_key('log_fold')):
		log_fold = root_path + '/result/' + kw['log_fold']
	if os.path.isdir(log_fold) == False:
		try:
			os.makedirs(log_fold)
		except:
			print "Can not create log fold! "
			return
	os.chdir(log_fold)

	results = []
	results.append(['Feat','Algorithm','Proportion', 'Accuracy', 'std'])
	log_file = 'feat_TD4'
	
	clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)
	# X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.8, random_state=random.randrange(1,51))

	# # proportion is 1.0
	# X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.8, random_state=random.randrange(1,51))
	# scores = clf.fit(X_train, y_train).score(X_test, y_test)
	# results.append(['feat_TD4', "LDA(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)", 1, scores.mean(), scores.std()])

	

	iteration = 10
	while(iteration>0):
		# proportion is 1.0  cv=5
		X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.8, random_state=random.randrange(1,51))
		scores = sklearn.cross_validation.cross_val_score(clf, trains, classes, cv=5)
		results.append(['feat_TD4', "LDA(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)", 1, scores.mean(), scores.std()])

		# proportion is 0.9
		X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
		trains_noise_1, classes_noise_1 = proportion_simu(X_train, y_train, 0.9)
		scores = clf.fit(trains_noise_1, classes_noise_1).score(X_test, y_test)
		results.append(['feat_TD4', clf, '0.9', scores.mean(), scores.std()])

		# proportion is 0.8
		X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
		trains_noise_2, classes_noise_2 = proportion_simu(X_train, y_train, 0.8)
		scores = clf.fit(trains_noise_2, classes_noise_2).score(X_test, y_test)
		results.append(['feat_TD4', clf, '0.8', scores.mean(), scores.std()])

		# proportion is 0.8 + 0.9
		trains_noise_3 = np.concatenate((trains_noise_1, trains_noise_2), axis=0)
		classes_noise_3 = np.concatenate((classes_noise_1, classes_noise_2), axis=0)
		scores = clf.fit(trains_noise_3, classes_noise_3).score(X_test, y_test)
		results.append(['feat_TD4', clf, '0.8+0.9', scores.mean(), scores.std()])

		# proportion is 0.7
		X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
		trains_noise, classes_noise = proportion_simu(X_train, y_train, 0.7)
		scores = clf.fit(trains_noise, classes_noise).score(X_test, y_test)
		results.append(['feat_TD4', clf, '0.7', scores.mean(), scores.std()])

		# proportion is 0.6
		X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
		trains_noise, classes_noise = proportion_simu(X_train, y_train, 0.6)
		scores = clf.fit(trains_noise, classes_noise).score(X_test, y_test)
		results.append(['feat_TD4', clf, '0.6', scores.mean(), scores.std()])

		iteration -= 1

	log_result(results, log_file+'_without_noise', 2)
	print 'training TD4 time elapsed:', time.time() - start_time


def training_with_pp(my_clfs, trains, classes, combine_feat, **kw):
	cv = 5
	if(kw.has_key('log_fold')):
		log_fold = root_path + '/result/' + kw['log_fold']
		if not new_fold(log_fold):
			return
	os.chdir(log_fold)

	results = []
	results.append(['Feat','Algorithm', 'Accuracy', 'std'])
	log_file = 'feat_1_pp'
	
	# tuple of all parallel python servers to connect with
	ppservers = ()
	#ppservers = ("10.0.0.1",)
	if len(sys.argv) > 1:
	    ncpus = int(sys.argv[1])
	    # Creates jobserver with ncpus workers
	    job_server = pp.Server(ncpus, ppservers=ppservers)
	else:
	    # Creates jobserver with automatically detected number of workers
	    job_server = pp.Server(ppservers=ppservers)

	start_time = time.time()
	jobs = []
	for i in range(len(feat_map)):
		if  i == 12 or i == 13 or i == 14 or i == 15:
			continue
		jobs.append((job_server.submit(myclassifys,(my_clfs, i, trains[i], classes, cv), (myclassify,), import_module)))

	for job in jobs:
		result = job()
		results.extend(result)
	print 'Classify time elapsed:', time.time() - start_time, 's'
	log_result(results, log_file, 2)


def myclassifys(my_clfs, feat_num, train, classes, cv):
	results = []
	for clf in my_clfs:
		scores = myclassify(clf, train, classes, cv)
		results.append([feat_num, clf, 1-scores.mean(), scores.std()])
	return results

def myclassify(my_clf, train, classes, cv):
	
	if(my_clf == 'LDA'):
		clf = eval('sklearn.lda.'+ my_clf)(solver='svd', shrinkage=None, priors=None, n_components=4, store_covariance=False, tol=0.0001)
		if(cv != 0):
			scores = sklearn.cross_validation.cross_val_score(clf, train, classes, cv=cv)
		else:
			X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(train, classes, test_size=0.6, random_state=0)
			scores = clf.fit(X_train, y_train).score(X_test, y_test)
			# print X_train.shape, y_train.shape, X_test.shape, y_test.shape
			# print clf.fit(X_train, y_train).predict_proba(X_test)

	if(my_clf == 'QDA'):
		clf = eval('sklearn.qda.'+ my_clf)()
		scores = sklearn.cross_validation.cross_val_score(clf, train, classes, cv=cv)
	
	if(my_clf == 'GaussianNB'):
		clf = eval('sklearn.naive_bayes.'+ my_clf)()
		scores = sklearn.cross_validation.cross_val_score(clf, train, classes, cv=cv)

	if(my_clf == 'SVC_linear'):
		clf = sklearn.svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
			intercept_scaling=1, loss='squared_hinge', multi_class='ovr', penalty='l2',
			random_state=None, tol=0.0001, verbose=0)
		scores = sklearn.cross_validation.cross_val_score(clf, train, classes, cv=cv)

	if(my_clf == 'SVC_rbf'):
		clf = sklearn.svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
			gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
			shrinking=True, tol=0.001, verbose=False)
		scores = sklearn.cross_validation.cross_val_score(clf, train, classes, cv=cv)

	if(my_clf == 'Logistic'):
		pen=['l1','l2']
		C=[1e5,1.0,10.0,0.1,1e-5]
		clf = sklearn.linear_model.LogisticRegression(C=0.5, penalty='l1')
		scores = sklearn.cross_validation.cross_val_score(clf, train, classes, cv=cv)
		for i in pen:
			for c in C:
				clf = sklearn.linear_model.LogisticRegression(C=c, penalty=i)
				temp_scores = sklearn.cross_validation.cross_val_score(clf, train, classes, cv=cv)
				if(scores.mean() < temp_scores.mean()):
					scores = temp_scores
	return scores

