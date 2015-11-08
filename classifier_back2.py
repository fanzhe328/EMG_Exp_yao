#!/usr/lib/python
import numpy as np
import os
import time
import sys
import random
import sklearn
import sklearn.lda
import sklearn.qda
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
import_module = ("numpy as np", "sklearn.cross_validation",
                 "sklearn.lda", "sklearn.qda", "sklearn.naive_bayes", "sklearn.svm")


def log_result(results, log_file, flag):
    np.save(log_file + '.npy', results)
    if flag == 2:
        np.savetxt(log_file + '.csv', results, fmt="%s", delimiter=",")


def new_fold(log_fold):
    if os.path.isdir(log_fold) == False:
        try:
            os.makedirs(log_fold)
        except:
            print "Can not create log fold! "
            return False
    return True


def training_lda_signal(my_clfs, trains, classes, **kw):
    start_time = time.time()
    print "training lda signal............."
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
    results.append(['Feat', 'Algorithm', 'Proportion', 'Accuracy', 'std'])
    log_file = 'signal'

    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False, tol=0.0001)
    # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.8, random_state=random.randrange(1,51))

    # test_rate_list = [i for i in range(1,9)]
    test_rate_list = [0.2, 0.4, 0.6, 0.8]

    # proportion is 1.0  cv=5
    # scores = sklearn.cross_validation.cross_val_score(clf, trains, classes, cv=10)
    # results.append(['signal_cv_10', 'lda(svd,tol=0.0001)', 1.0, scores.mean(), scores.std()])

    for i in test_rate_list:

        iteration = 9

        scores_1_0 = np.zeros((iteration + 1,))
        scores_0_9 = np.zeros((iteration + 1,))
        scores_0_8 = np.zeros((iteration + 1,))
        scores_0_89 = np.zeros((iteration + 1,))

        while(iteration >= 0):
            X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
                trains, classes, test_size=i, random_state=random.randrange(1, 51))
            print 'test_size:', i
            print X_train.shape, X_test.shape, y_train.shape, y_test.shape

            # proportion is 1.0
            scores = clf.fit(X_train, y_train).score(X_test, y_test)
            scores_1_0[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', 1.0, scores.mean(), scores.std()])

            # proportion is 0.9
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            trains_noise_1, classes_noise_1 = proportion_simu(
                X_train, y_train, 0.9)
            scores = clf.fit(trains_noise_1, classes_noise_1).score(
                X_test, y_test)
            scores_0_9[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.9', scores.mean(), scores.std()])

            # proportion is 0.8
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            trains_noise_2, classes_noise_2 = proportion_simu(
                X_train, y_train, 0.8)

            scores = clf.fit(trains_noise_2, classes_noise_2).score(
                X_test, y_test)
            scores_0_8[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8', scores.mean(), scores.std()])

            # # proportion is 0.8 + 0.9
            trains_noise_3 = np.concatenate(
                (trains_noise_1, trains_noise_2), axis=0)
            classes_noise_3 = np.concatenate(
                (classes_noise_1, classes_noise_2), axis=0)
            scores = clf.fit(trains_noise_3, classes_noise_3).score(
                X_test, y_test)
            scores_0_89[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8+0.9', scores.mean(), scores.std()])

            iteration -= 1
        results.append(['signal', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '1.0', np.mean(scores_1_0), np.std(scores_1_0)])
        results.append(['signal', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '0.9', np.mean(scores_0_9), np.std(scores_0_9)])
        results.append(['signal', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '0.8', np.mean(scores_0_8), np.std(scores_0_8)])
        results.append(['signal', 'lda(svd,tol=0.0001,test_rate=' + str(i) + ')',
                        '0.8+0.9', np.mean(scores_0_89), np.std(scores_0_89)])

    log_result(results, log_file + '_' + str(kw['num']), 2)
    print 'training TD4 time elapsed:', time.time() - start_time


def training_lda_TD4(my_clfs, trains, classes, **kw):
    start_time = time.time()
    print "----training TD4............."
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
    results.append(['Feat', 'Algorithm', 'Proportion', 'Accuracy', 'std'])
    log_file = 'feat_TD4'

    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False, tol=0.0001)
    # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.8, random_state=random.randrange(1,51))

    # # proportion is 1.0
    # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.8, random_state=random.randrange(1,51))
    # scores = clf.fit(X_train, y_train).score(X_test, y_test)
    # results.append(['feat_TD4', "LDA(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)", 1, scores.mean(), scores.std()])

    # test_rate_list = [i for i in range(1,9)]
    test_rate_list = [0.2, 0.4, 0.6, 0.8]

    # proportion is 1.0  cv=5
    scores = sklearn.cross_validation.cross_val_score(
        clf, trains, classes, cv=10)
    results.append(['feat_TD4_cv_5', 'lda(svd,tol=0.0001)',
                    1.0, scores.mean(), scores.std()])

    for i in test_rate_list:

        iteration = 10

        scores_1_0 = np.zeros((iteration + 1,))
        scores_1_1 = np.zeros((iteration + 1,))
        scores_1_2 = np.zeros((iteration + 1,))
        scores_1_12 = np.zeros((iteration + 1,))
        scores_0_9 = np.zeros((iteration + 1,))
        scores_0_8 = np.zeros((iteration + 1,))
        scores_0_89 = np.zeros((iteration + 1,))
        scores_19 = np.zeros((iteration + 1,))
        scores_1289 = np.zeros((iteration + 1,))

        while(iteration >= 0):
            X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
                trains, classes, test_size=i, random_state=random.randrange(1, 51))
            print 'test_size:', i
            print X_train.shape, X_test.shape, y_train.shape, y_test.shape

            # proportion is 1.0
            scores = clf.fit(X_train, y_train).score(X_test, y_test)
            scores_1_0[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', 1.0, scores.mean(), scores.std()])

            # proportion is 1.1
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            trains_noise_1, classes_noise_1 = proportion_simu(
                X_train, y_train, 0.9)
            scores = clf.fit(trains_noise_1, classes_noise_1).score(
                X_test, y_test)
            scores_1_1[iteration] = scores.mean()

            # proportion is 1.2
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            trains_noise_2, classes_noise_2 = proportion_simu(
                X_train, y_train, 0.8)
            scores = clf.fit(trains_noise_2, classes_noise_2).score(
                X_test, y_test)
            scores_1_2[iteration] = scores.mean()

            # # proportion is 1.1 + 1.2
            trains_noise_12 = np.concatenate(
                (trains_noise_1, trains_noise_2), axis=0)
            classes_noise_12 = np.concatenate(
                (classes_noise_1, classes_noise_2), axis=0)
            scores = clf.fit(trains_noise_12, classes_noise_12).score(
                X_test, y_test)
            scores_1_12[iteration] = scores.mean()

            # proportion is 0.9
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            trains_noise_9, classes_noise_9 = proportion_simu(
                X_train, y_train, 0.9)
            scores = clf.fit(trains_noise_9, classes_noise_9).score(
                X_test, y_test)
            scores_0_9[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.9', scores.mean(), scores.std()])

            # proportion is 0.8
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            trains_noise_8, classes_noise_8 = proportion_simu(
                X_train, y_train, 0.8)
            scores = clf.fit(trains_noise_8, classes_noise_8).score(
                X_test, y_test)
            scores_0_8[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8', scores.mean(), scores.std()])

            # # proportion is 0.8 + 0.9
            trains_noise_89 = np.concatenate(
                (trains_noise_8, trains_noise_9), axis=0)
            classes_noise_89 = np.concatenate(
                (classes_noise_8, classes_noise_9), axis=0)
            scores = clf.fit(trains_noise_89, classes_noise_89).score(
                X_test, y_test)
            scores_0_89[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8+0.9', scores.mean(), scores.std()])

            # # proportion is 1.1 + 0.9
            trains_noise_19 = np.concatenate(
                (trains_noise_1, trains_noise_9), axis=0)
            classes_noise_19 = np.concatenate(
                (classes_noise_1, classes_noise_9), axis=0)
            scores = clf.fit(trains_noise_19, classes_noise_19).score(
                X_test, y_test)
            scores_19[iteration] = scores.mean()

            # # proportion is 1.1 + 0.9 + 1.2 + 0.8
            trains_noise_1289 = np.concatenate(
                (trains_noise_12, trains_noise_89), axis=0)
            classes_noise_1289 = np.concatenate(
                (classes_noise_12, classes_noise_89), axis=0)
            scores = clf.fit(trains_noise_1289,
                             classes_noise_1289).score(X_test, y_test)
            scores_1289[iteration] = scores.mean()

            # # proportion is 0.7
            # # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            # trains_noise, classes_noise = proportion_simu(X_train, y_train, 0.7)
            # scores = clf.fit(trains_noise, classes_noise).score(X_test, y_test)
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.7', scores.mean(), scores.std()])

            # # proportion is 0.6
            # # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            # trains_noise, classes_noise = proportion_simu(X_train, y_train, 0.6)
            # scores = clf.fit(trains_noise, classes_noise).score(X_test, y_test)
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.6', scores.mean(), scores.std()])

            iteration -= 1
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '1.0', np.mean(scores_1_0), np.std(scores_1_0)])
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '0.9', np.mean(scores_0_9), np.std(scores_0_9)])
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '0.8', np.mean(scores_0_8), np.std(scores_0_8)])
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' + str(i) + ')',
                        '0.8+0.9', np.mean(scores_0_89), np.std(scores_0_89)])
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '1.1', np.mean(scores_1_1), np.std(scores_1_1)])
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '1.2', np.mean(scores_1_2), np.std(scores_1_2)])
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '1.1+0.9', np.mean(scores_19), np.std(scores_19)])
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' + str(i) + ')',
                        '1.1+0.9+1.2+0.8', np.mean(scores_1289), np.std(scores_1289)])

    log_result(results, log_file + '_' + str(kw['num']), 2)
    print '----training TD4 time elapsed:', time.time() - start_time


def training_lda_TD4_cross(my_clfs, X_train, y_train, X_test, y_test, **kw):
    start_time = time.time()
    print "training TD4 cross............."

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
    results.append(['Feat', 'Algorithm', 'Proportion', 'Accuracy', 'std'])
    log_file = 'feat_TD4'
    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False, tol=0.0001)
    test_rate_list = [0.2, 0.4, 0.6, 0.8]

    for i in test_rate_list:

        iteration = 4

        scores_1_0 = np.zeros((iteration + 1,))
        scores_0_9 = np.zeros((iteration + 1,))
        scores_0_8 = np.zeros((iteration + 1,))
        scores_0_89 = np.zeros((iteration + 1,))

        while(iteration >= 0):
            print 'test_rate:', i, 'iteration:', iteration
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=i, random_state=random.randrange(1,51))
            # print 'test_size:', i
            print X_train.shape, X_test.shape, y_train.shape, y_test.shape

            # proportion is 1.0
            scores = clf.fit(X_train, y_train).score(X_test, y_test)
            scores_1_0[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', 1.0, scores.mean(), scores.std()])

            # proportion is 0.9
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            trains_noise_1, classes_noise_1 = proportion_simu(
                X_train, y_train, 0.9)
            scores = clf.fit(trains_noise_1, classes_noise_1).score(
                X_test, y_test)
            scores_0_9[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.9', scores.mean(), scores.std()])

            # proportion is 0.8
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            trains_noise_2, classes_noise_2 = proportion_simu(
                X_train, y_train, 0.8)

            scores = clf.fit(trains_noise_2, classes_noise_2).score(
                X_test, y_test)
            scores_0_8[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8', scores.mean(), scores.std()])

            # # proportion is 0.8 + 0.9
            trains_noise_3 = np.concatenate(
                (trains_noise_1, trains_noise_2), axis=0)
            classes_noise_3 = np.concatenate(
                (classes_noise_1, classes_noise_2), axis=0)
            scores = clf.fit(trains_noise_3, classes_noise_3).score(
                X_test, y_test)
            scores_0_89[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8+0.9', scores.mean(), scores.std()])

            iteration -= 1
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '1.0', np.mean(scores_1_0), np.std(scores_1_0)])
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '0.9', np.mean(scores_0_9), np.std(scores_0_9)])
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' +
                        str(i) + ')', '0.8', np.mean(scores_0_8), np.std(scores_0_8)])
        results.append(['feat_TD4', 'lda(svd,tol=0.0001,test_rate=' + str(i) + ')',
                        '0.8+0.9', np.mean(scores_0_89), np.std(scores_0_89)])

    log_result(results, log_file + '_' + str(kw['num']), 2)
    print 'training TD4 time elapsed:', time.time() - start_time


def training_with_pp(my_clfs, trains, classes, combine_feat, **kw):
    cv = 5
    if(kw.has_key('log_fold')):
        log_fold = root_path + '/result/' + kw['log_fold']
        if not new_fold(log_fold):
            return
    os.chdir(log_fold)

    results = []
    results.append(['Feat', 'Algorithm', 'Accuracy', 'std'])
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
        if i == 12 or i == 13 or i == 14 or i == 15:
            continue
        jobs.append((job_server.submit(myclassifys, (my_clfs, i, trains[
                    i], classes, cv), (myclassify,), import_module)))

    for job in jobs:
        result = job()
        results.extend(result)
    print 'Classify time elapsed:', time.time() - start_time, 's'
    log_result(results, log_file, 2)


def myclassifys(my_clfs, feat_num, train, classes, cv):
    results = []
    for clf in my_clfs:
        scores = myclassify(clf, train, classes, cv)
        results.append([feat_num, clf, 1 - scores.mean(), scores.std()])
    return results


def myclassify(my_clf, train, classes, cv):

    if(my_clf == 'LDA'):
        clf = eval('sklearn.lda.' + my_clf)(solver='svd', shrinkage=None,
                                            priors=None, n_components=4, store_covariance=False, tol=0.0001)
        if(cv != 0):
            scores = sklearn.cross_validation.cross_val_score(
                clf, train, classes, cv=cv)
        else:
            X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
                train, classes, test_size=0.6, random_state=0)
            scores = clf.fit(X_train, y_train).score(X_test, y_test)
            # print X_train.shape, y_train.shape, X_test.shape, y_test.shape
            # print clf.fit(X_train, y_train).predict_proba(X_test)

    if(my_clf == 'QDA'):
        clf = eval('sklearn.qda.' + my_clf)()
        scores = sklearn.cross_validation.cross_val_score(
            clf, train, classes, cv=cv)

    if(my_clf == 'GaussianNB'):
        clf = eval('sklearn.naive_bayes.' + my_clf)()
        scores = sklearn.cross_validation.cross_val_score(
            clf, train, classes, cv=cv)

    if(my_clf == 'SVC_linear'):
        clf = sklearn.svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                                    intercept_scaling=1, loss='squared_hinge', multi_class='ovr', penalty='l2',
                                    random_state=None, tol=0.0001, verbose=0)
        scores = sklearn.cross_validation.cross_val_score(
            clf, train, classes, cv=cv)

    if(my_clf == 'SVC_rbf'):
        clf = sklearn.svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                              gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
                              shrinking=True, tol=0.001, verbose=False)
        scores = sklearn.cross_validation.cross_val_score(
            clf, train, classes, cv=cv)

    if(my_clf == 'Logistic'):
        pen = ['l1', 'l2']
        C = [1e5, 1.0, 10.0, 0.1, 1e-5]
        clf = sklearn.linear_model.LogisticRegression(C=0.5, penalty='l1')
        scores = sklearn.cross_validation.cross_val_score(
            clf, train, classes, cv=cv)
        for i in pen:
            for c in C:
                clf = sklearn.linear_model.LogisticRegression(C=c, penalty=i)
                temp_scores = sklearn.cross_validation.cross_val_score(
                    clf, train, classes, cv=cv)
                if(scores.mean() < temp_scores.mean()):
                    scores = temp_scores
    return scores
