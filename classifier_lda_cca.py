# *-* coding=utf-8 *-*
#!/usr/lib/python
import os, sys, time, random
import numpy as np
import sklearn
import sklearn.lda
import sklearn.qda
import sklearn.naive_bayes
import sklearn.cross_validation
import sklearn.svm
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
# from noise_simulation import proportion_simu
# from sklearn import linear_model
# from sklearn.lda import LDA
# from sklearn.qda import QDA
# from sklearn.svm import SVC, LinearSVC
# from sklearn.naive_bayes import GaussianNB

root_path = os.getcwd()
transform_fold = root_path + '/result/transform'
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


'''
    个体内训练策略（针对同一个受试者的）
'''
def training_lda_TD4_intra(my_clfs, trains, classes, **kw):

    start_time = time.time()
    if(kw.has_key('log_fold')):
        log_fold = root_path + '/result/' + kw['log_fold']
    new_fold(log_fold)

    chan_len = kw['chan_len']
    action_num = kw['action_num']
    cv = 5
    results = []
    results.append(
        ['Feat', 'Algorithm','n_components', 'Channel_Pos', 'Accuracy', 'std'])
    log_file = 'feat_'+kw['feature_type']+'_intra'

    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False,
                          tol=0.0001)

    data_num = trains.shape[0]/action_num

    scores = sklearn.cross_validation.cross_val_score(clf, trains, classes, cv=cv)
    results.append(['feat_TD4_cv_'+str(cv), 'lda', 'ALL', 0, scores.mean(), scores.std()])
    
    # 子空间维数
    n_components_list = [6, 8, 10, 12, 14, 16]

    # PLS CCA
    plsca = PLSCanonical(n_components=15)

    for n_components in n_components_list:

        for idx, channel_pos in enumerate(kw['pos_list']):
            print '----training TD4 intra CCA based, channel_pos: ', channel_pos,'......'
            if channel_pos == 'S0':
                trains_intra_S0 = trains[:,idx*chan_len: idx*chan_len+chan_len]
            elif channel_pos != 'S0':
                trains_intra_shift = trains[:,idx*chan_len: idx*chan_len+chan_len]
                # m = trains_intra_S0.shape[0]
                plsca.fit(trains_intra_shift, trains_intra_S0)
                trains_intra = plsca.transform(trains_intra_shift)

                scores = sklearn.cross_validation.cross_val_score(
                    clf, trains_intra, classes, cv=cv)
                results.append(['feat_TD4_cv_'+str(cv), 'lda_cca', n_components, 'S0+'+channel_pos, scores.mean(), scores.std()])

        for idx, channel_pos in enumerate(kw['pos_list']):
            print '----training TD4 intra , channel_pos: ', channel_pos,'......'
            trains_intra = trains[:,idx*chan_len: idx*chan_len+chan_len]

            scores = sklearn.cross_validation.cross_val_score(
                clf, trains_intra, classes, cv=cv)
            results.append(['feat_TD4_cv_'+str(cv), 'lda', n_components, channel_pos, scores.mean(), scores.std()])

        for idx, channel_pos in enumerate(kw['pos_list']):
            print '----training TD4 intra combination, channel_pos: ', channel_pos,'......'
            trains_intra_shift = trains[:,idx*chan_len: idx*chan_len+chan_len]
            trains_intra = np.concatenate( (trains_intra_S0, trains_intra_shift), axis=0 )
            classes_intra = np.concatenate( (classes, classes), axis=0)
            # print trains_intra_shift.shape, trains_intra.shape, classes_intra.shape
            # sys.exit(0)
            scores = sklearn.cross_validation.cross_val_score(
                clf, trains_intra, classes_intra, cv=cv)
            results.append(['feat_TD4_cv_'+str(cv), 'lda', n_components, channel_pos+' combine S0', scores.mean(), scores.std()])
    
    # print results
    # sys.exit(0)
    log_result(results, log_fold + '/' + log_file + '_action_1-'+str(action_num), 2)
    print '----Log Fold:', log_fold, ', log_file: ', log_file + '_action_1-'+str(action_num)
    print '----training TD4 time elapsed:', time.time() - start_time


def generate_transform_equations(trains_S0, trains_shift, **kw):
    print 'generate transform equations.........'
    new_fold(transform_fold)
    chan_len = kw['chan_len']
    for idx, channel_pos in enumerate(kw['pos_list']):
        X_trains = trains_shift[:,idx*chan_len:idx*chan_len+chan_len]
        plsca = PLSCanonical(n_components=12)
        plsca.fit(X_trains, trains_S0)
        joblib.dump(plsca, transform_fold+'/cca_transform_'+kw['subject']+'_'+channel_pos+'.model')
        # log_result(plsca, transform_fold+'/cca_transform_'+kw['subject']+'_'+channel_pos, 1)


def training_lda_TD4_inter(my_clfs, trains_S0, trains_shift, classes, **kw):
    print 'training_lda_TD4_inter.........'
    start_time = time.time()
    if(kw.has_key('log_fold')):
        log_fold = root_path + '/result/' + kw['log_fold']
    new_fold(log_fold)

    chan_len = kw['chan_len']

    print "----training "+kw['feature_type']+" inter, training by position O, testing by electrode shift ", 

    cv = 5
    results = []
    results.append(
        ['Feat', 'Algorithm','Channel_Pos', 'Accuracy', 'std'])
    log_file = 'feat_'+kw['feature_type']+'_inter'

    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False,
                          tol=0.0001)

    data_num = trains_S0.shape[0]/kw['action_num']
    # print data_num
    
    scores = sklearn.cross_validation.cross_val_score(
        clf, trains_S0, classes, cv=cv)
    results.append(['feat_TD4_cv_'+str(cv), 'lda', 'S0',
                    scores.mean(), scores.std()])
    
    kf = KFold(data_num, n_folds=cv)
    
    for idx, channel_pos in enumerate(kw['pos_list']):
        print 'channel_pos: ', channel_pos,'......'

        X_test = trains_shift[:,idx*chan_len:idx*chan_len+chan_len]
        y_test = classes

        iteration = cv
        scores = np.zeros((iteration,))
        cca_scores = np.zeros((iteration,))
        
        # print type(scores[0])
        # sys.exit(0)
        iteration -= 1
        for train_idx, test_idx in kf:
            train_idx = np.concatenate( (train_idx,train_idx+data_num,train_idx+data_num*2,
                                        train_idx+data_num*3,train_idx+data_num*4,train_idx+data_num*5,
                                        train_idx+data_num*6), axis=None)
            X_train, y_train = trains_S0[train_idx], classes[train_idx]

            X_train_shift, y_train_shift = X_test[train_idx], classes[train_idx]
            X_train_all = np.concatenate( (X_train, X_train_shift), axis=0)
            y_train_all = np.concatenate( (y_train, y_train_shift), axis=0)
            # print X_train_all.shape
            # sys.exit(0)
            score_inter = clf.fit(X_train_all, y_train_all).score(X_test, y_test)
            scores[iteration] = score_inter.mean()
            # print X_train.shape, y_train.shape
            

            if channel_pos != 'S0':

                # plsca = joblib.load(transform_fold+'/cca_transform_'+kw['subject']+'_'+channel_pos+'.model')
                plsca = PLSCanonical(n_components=14)
                # print X_test.shape, X_train.shape
                # sys.exit(0)
                plsca.fit(X_test[train_idx], X_train)
                X_test_cca, X_train_cca = plsca.transform(X_test, X_train)
                cca_score = clf.fit(X_train_cca, y_train).score(X_test_cca, y_test)
                cca_scores[iteration] = cca_score.mean()

            iteration -= 1

        # print scores
        # print cca_scores
        # sys.exit(0)
        results.append(['feat_TD4', 'lda', 
                        channel_pos, np.mean(scores), np.std(scores)])
        results.append(['feat_TD4', 'lda_cca', 
                        channel_pos, np.mean(cca_scores), np.std(cca_scores)])

            
    
    
    log_result(results, log_fold + '/' + log_file + '_' + str(kw['num']), 2)
    print '----Log Fold:', log_fold, ', log_file: ', log_file + '_' + channel_pos + '_' + str(kw['num'])
    print '----training TD4 time elapsed:', time.time() - start_time

    

    # mean_shift = 0
    # std_shift = 0
    # for i in range(2, 10):
    #     mean_shift += results[i][4]
    #     std_shift += results[i][5]
    # mean_shift /= 9
    # std_shift /= 9

    # results.append(['feat_TD4','lda(svd;tol=0.0001)', 'Shift_means', '1.0', mean_shift, std_shift])

    # mean_all = 0
    # std_all = 0
    # for i in range(1, 10):
    #     mean_all += results[i][4]
    #     std_all += results[i][5]
    # mean_all /= 9
    # std_all /= 9