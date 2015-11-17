#!/usr/lib/python
import os, sys, time, random
import numpy as np
import sklearn
import sklearn.lda
import sklearn.qda
import sklearn.naive_bayes
import sklearn.cross_validation
import sklearn.svm

from sklearn.cross_validation import KFold
from noise_simulation import proportion_simu

# from sklearn import linear_model
# from sklearn.lda import LDA
# from sklearn.qda import QDA
# from sklearn.svm import SVC, LinearSVC
# from sklearn.naive_bayes import GaussianNB

root_path = os.getcwd()
import_module = ("numpy as np", "sklearn.cross_validation",
                 "sklearn.lda", "sklearn.qda", "sklearn.naive_bayes", 
                 "sklearn.svm")


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


def training_lda_TD4_intra(my_clfs, trains, classes, **kw):
    start_time = time.time()
    if(kw.has_key('log_fold')):
        log_fold = root_path + '/result/' + kw['log_fold']
    new_fold(log_fold)

    chan_len = kw['chan_len']
    cv = 5
    results = []
    results.append(
        ['Feat', 'Algorithm','Channel_Pos', 'Accuracy', 'std'])
    log_file = 'feat_'+kw['feature_type']+'_intra'

    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False,
                          tol=0.0001)


    test_rate_list = [0.2]
    data_num = trains.shape[0]/kw['action_num']

    for idx, channel_pos in enumerate(kw['pos_list']):
        print '----training TD4 intra, channel_pos: ', channel_pos,'......'
        trains_intra = trains[:,idx*chan_len: idx*chan_len+chan_len]

        scores = sklearn.cross_validation.cross_val_score(
            clf, trains_intra, classes, cv=cv)
        results.append(['feat_TD4_cv_'+str(cv), 'lda(svd;tol=0.0001)', channel_pos,
                         scores.mean(), scores.std()])

    log_result(results, log_fold + '/' + log_file + '_' + str(kw['num']), 2)
    print '----Log Fold:', log_fold, ', log_file: ', log_file + '_' + str(kw['num'])
    print '----training TD4 time elapsed:', time.time() - start_time

def training_lda_TD4_inter(my_clfs, trains, trains_classes, tests, tests_classes, **kw):
    start_time = time.time()
    if(kw.has_key('log_fold')):
        log_fold = root_path + '/result/' + kw['log_fold']
    new_fold(log_fold)

    chan_len = kw['chan_len']

    print "----training TD4 inter, training by position O, testing by electrode shift ", 

    cv = 5
    results = []
    results.append(
        ['Feat', 'Algorithm','Channel_Pos', 'Proportion', 'Accuracy', 'std'])
    log_file = 'feat_'+kw['feature_type']+'_inter'

    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False,
                          tol=0.0001)
    test_rate_list = [0.2]
    data_num = trains.shape[0]/kw['action_num']
    # print data_num
    
    scores = sklearn.cross_validation.cross_val_score(
        clf, trains, trains_classes, cv=cv)
    results.append(['feat_TD4_cv_'+str(cv), 'lda(svd;tol=0.0001)', 'S0',
                    1, scores.mean(), scores.std()])
    
    kf = KFold(data_num, n_folds=cv)
    
    for idx, channel_pos in enumerate(kw['pos_list']):
        print 'channel_pos: ', channel_pos,'......'
        X_test = tests[:,idx*chan_len:idx*chan_len+chan_len]
        y_test = tests_classes

        iteration = cv
        scores_1_0 = np.zeros((iteration,))
        # scores_1_1 = np.zeros((iteration,))
        # scores_1_2 = np.zeros((iteration,))
        # scores_1_12 = np.zeros((iteration,))
        # scores_0_9 = np.zeros((iteration,))
        # scores_0_8 = np.zeros((iteration,))
        # scores_0_89 = np.zeros((iteration,))
        # scores_19 = np.zeros((iteration,))
        # scores_1289 = np.zeros((iteration,))
        iteration -= 1
        for train_idx, test_idx in kf:
            train_idx = np.concatenate( (train_idx,train_idx+data_num,train_idx+data_num*2,
                                        train_idx+data_num*3,train_idx+data_num*4,train_idx+data_num*5,
                                        train_idx+data_num*6), axis=None)
            X_train, y_train = trains[train_idx], trains_classes[train_idx]
            # print X_train.shape, y_train.shape
            # sys.exit(0)
            # print '--iteration: ', str(5-iteration),'----'

            # proportion is 1.0
            scores = clf.fit(X_train, y_train).score(X_test, y_test)
            scores_1_0[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', 1.0, scores.mean(), scores.std()])

            # proportion is 1.1
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            # trains_noise_1, classes_noise_1 = proportion_simu(
            #     X_train, y_train, 0.9)
            # scores = clf.fit(trains_noise_1, classes_noise_1).score(
            #     X_test, y_test)
            # scores_1_1[iteration] = scores.mean()

            # proportion is 1.2
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            # trains_noise_2, classes_noise_2 = proportion_simu(
            #     X_train, y_train, 0.8)
            # scores = clf.fit(trains_noise_2, classes_noise_2).score(
            #     X_test, y_test)
            # scores_1_2[iteration] = scores.mean()

            # # proportion is 1.1 + 1.2
            # trains_noise_12 = np.concatenate(
            #     (trains_noise_1, trains_noise_2), axis=0)
            # classes_noise_12 = np.concatenate(
            #     (classes_noise_1, classes_noise_2), axis=0)
            # scores = clf.fit(trains_noise_12, classes_noise_12).score(
            #     X_test, y_test)
            # scores_1_12[iteration] = scores.mean()

            # proportion is 0.9
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            # trains_noise_9, classes_noise_9 = proportion_simu(
            #     X_train, y_train, 0.9)
            # scores = clf.fit(trains_noise_9, classes_noise_9).score(
            #     X_test, y_test)
            # scores_0_9[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.9', scores.mean(), scores.std()])

            # proportion is 0.8
            # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
            # trains_noise_8, classes_noise_8 = proportion_simu(
            #     X_train, y_train, 0.8)
            # scores = clf.fit(trains_noise_8, classes_noise_8).score(
            #     X_test, y_test)
            # scores_0_8[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8', scores.mean(), scores.std()])

            # # proportion is 0.8 + 0.9
            # trains_noise_89 = np.concatenate(
            #     (trains_noise_8, trains_noise_9), axis=0)
            # classes_noise_89 = np.concatenate(
            #     (classes_noise_8, classes_noise_9), axis=0)
            # scores = clf.fit(trains_noise_89, classes_noise_89).score(
            #     X_test, y_test)
            # scores_0_89[iteration] = scores.mean()
            # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8+0.9', scores.mean(), scores.std()])

            # # proportion is 1.1 + 0.9
            # trains_noise_19 = np.concatenate(
            #     (trains_noise_1, trains_noise_9), axis=0)
            # classes_noise_19 = np.concatenate(
            #     (classes_noise_1, classes_noise_9), axis=0)
            # scores = clf.fit(trains_noise_19, classes_noise_19).score(
            #     X_test, y_test)
            # scores_19[iteration] = scores.mean()

            # # proportion is 1.1 + 0.9 + 1.2 + 0.8
            # trains_noise_1289 = np.concatenate(
            #     (trains_noise_12, trains_noise_89), axis=0)
            # classes_noise_1289 = np.concatenate(
            #     (classes_noise_12, classes_noise_89), axis=0)
            # scores = clf.fit(trains_noise_1289,
            #                  classes_noise_1289).score(X_test, y_test)
            # scores_1289[iteration] = scores.mean()

            iteration -= 1
        results.append(['feat_TD4', 'lda(svd;tol=0.0001)', 
                        channel_pos, '1.0', np.mean(scores_1_0), np.std(scores_1_0)])
        # results.append(['feat_TD4', 'lda(svd;tol=0.0001)', 
        #                 channel_pos, '0.9', np.mean(scores_0_9), np.std(scores_0_9)])
        # results.append(['feat_TD4', 'lda(svd;tol=0.0001)', 
        #                 channel_pos, '0.8', np.mean(scores_0_8), np.std(scores_0_8)])
        # results.append(['feat_TD4', 'lda(svd;tol=0.0001)',
        #                 channel_pos, '0.8+0.9', np.mean(scores_0_89), np.std(scores_0_89)])
        # results.append(['feat_TD4', 'lda(svd;tol=0.0001)', 
        #                 channel_pos, '1.1', np.mean(scores_1_1), np.std(scores_1_1)])
        # results.append(['feat_TD4', 'lda(svd;tol=0.0001)', 
        #                 channel_pos, '1.2', np.mean(scores_1_2), np.std(scores_1_2)])
        # results.append(['feat_TD4', 'lda(svd;tol=0.0001)', 
        #                 channel_pos, '1.1+0.9', np.mean(scores_19), np.std(scores_19)])
        # results.append(['feat_TD4', 'lda(svd;tol=0.0001)', 
        #                 channel_pos,'1.1+0.9+1.2+0.8', np.mean(scores_1289), np.std(scores_1289)])
        #                 
    mean_shift = 0
    std_shift = 0
    for i in range(2, 10):
        mean_shift += results[i][4]
        std_shift += results[i][5]
    mean_shift /= 9
    std_shift /= 9

    results.append(['feat_TD4','lda(svd;tol=0.0001)', 'Shift_means', '1.0', mean_shift, std_shift])

    mean_all = 0
    std_all = 0
    for i in range(1, 10):
        mean_all += results[i][4]
        std_all += results[i][5]
    mean_all /= 9
    std_all /= 9

    results.append(['feat_TD4','lda(svd;tol=0.0001)', 'All_means', '1.0', mean_all, std_all])
    log_result(results, log_fold + '/' + log_file + '_' + str(kw['num']), 2)
    print '----Log Fold:', log_fold, ', log_file: ', log_file + '_' + channel_pos + '_' + str(kw['num'])
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
        results.append(['feat_TD4', 'lda(svd;tol=0.0001;test_rate=' +
                        str(i) + ')', '1.0', np.mean(scores_1_0), np.std(scores_1_0)])
        results.append(['feat_TD4', 'lda(svd;tol=0.0001;test_rate=' +
                        str(i) + ')', '0.9', np.mean(scores_0_9), np.std(scores_0_9)])
        results.append(['feat_TD4', 'lda(svd;tol=0.0001;test_rate=' +
                        str(i) + ')', '0.8', np.mean(scores_0_8), np.std(scores_0_8)])
        results.append(['feat_TD4', 'lda(svd;tol=0.0001;test_rate=' + str(i) + ')',
                        '0.8+0.9', np.mean(scores_0_89), np.std(scores_0_89)])

    log_result(results, log_file + '_' + str(kw['num']), 2)
    print 'training TD4 time elapsed:', time.time() - start_time