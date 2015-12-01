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
    cv = 4
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
    
    # 组内训练策略 9组数据
    print '组内训练.............'
    for idx, channel_pos in enumerate(kw['pos_list']):
        # print '----training TD4 intra , channel_pos: ', channel_pos,'......'
        trains_intra = trains[:,idx*chan_len: idx*chan_len+chan_len]

        scores = sklearn.cross_validation.cross_val_score(
            clf, trains_intra, classes, cv=cv)
        results.append(['feat_TD4_cv_'+str(cv), 'lda', 0, channel_pos, scores.mean(), scores.std()])


    # 中心训练策略
    print '中心训练策略.............'
    trains_intra_S0 = trains[:,0:chan_len]
    for idx, channel_pos in enumerate(kw['pos_list']):
        if channel_pos == 'S0':
            continue
        tests_shift = trains[:,idx*chan_len: idx*chan_len+chan_len]
        # if channel_pos == 'L2':
        #     print idx*chan_len, idx*chan_len+chan_len, tests_shift.shape, trains.shape
        #     sys.exit(0)
        scores = clf.fit(trains_intra_S0, classes).score(tests_shift, classes)
        results.append(['feat_TD4_cv_'+str(cv), 'lda', 0, 'train S0' + ' test ' + channel_pos, scores.mean(), scores.std()])
    
    # 组训练策略（不同于组内训练策略） 5-fold
    print '组训练策略.............'
    trains_intra_S0 = trains[:,0:chan_len]
    kf = KFold(data_num, n_folds=cv)
    for idx, channel_pos in enumerate(kw['pos_list']):
        if channel_pos == 'S0':
            continue
        itera = cv
        scores = np.zeros( (itera,) )
        # stds = np.zeros( (itera,) )
        itera -= 1
        trains_shift = trains[:,idx*chan_len: idx*chan_len+chan_len]
        for train_idx, test_idx in kf:
            train_idx_all = np.array([], np.int)
            test_idx_all = np.array([], np.int)

            for action_idx in range(action_num):
                train_idx_all = np.concatenate( (train_idx_all, train_idx*(action_idx+1)), axis=0)
                test_idx_all = np.concatenate( (test_idx_all, test_idx*(action_idx+1)), axis=0)

            X_train = np.concatenate( (trains_intra_S0[train_idx_all], trains_shift[train_idx_all]), axis=0)
            y_train = np.concatenate( (classes[train_idx_all], classes[train_idx_all]), axis=0)

            X_test = trains_shift[test_idx_all]
            y_test = classes[test_idx_all]

            # X_test = trains_shift
            # y_test = classes

            score = clf.fit(X_train, y_train).score(X_test, y_test)
            scores[itera] = score.mean()
            
            itera -= 1
        # print scores
        results.append(['feat_TD4_cv_'+str(cv), 'lda', 0, 'S0 + '+channel_pos, np.mean(scores), np.std(scores)])
    
    # 基于CCA的训练策略 5-fold 交叉验证
    print 'CCA训练策略.............'
    trains_S0 = trains[:,0:chan_len]
    n_components_list = [6, 8, 10, 12, 14, 16]              # 子空间维数
    # n_components_list = [12,14,16]
    kf = KFold(data_num, n_folds=cv)
    for n_components in n_components_list:
        
        for idx, channel_pos in enumerate(kw['pos_list']):
            if channel_pos == 'S0':
                continue
            itera = cv
            scores = np.zeros( (itera,) )
            stds = np.zeros( (itera,) )
            itera -= 1
            trains_shift = trains[:,idx*chan_len: idx*chan_len+chan_len]
            for train_idx, test_idx in kf:
                train_idx_all = np.array([], np.int)
                test_idx_all = np.array([], np.int)
                for action_idx in range(action_num):
                    train_idx_all = np.concatenate( (train_idx_all, train_idx*(action_idx+1)), axis=0)
                    test_idx_all = np.concatenate( (test_idx_all, test_idx*(action_idx+1)), axis=0)
                # print train_idx_all.shape, train_idx_all, test_idx_all.shape, test_idx_all
                # plsca.fit(trains_shift[train_idx_all], trains_S0[train_idx_all])
                plsca = PLSCanonical(n_components=n_components)
                plsca.fit(trains_shift, trains_S0)
                trains_shift_cca, trains_S0_cca = plsca.transform(trains_shift, trains_S0)
                X_trains = np.concatenate( (trains_S0_cca, trains_shift_cca[train_idx_all]), axis=0)
                y_trains = np.concatenate( (classes, classes[train_idx_all]), axis=0)
                score = clf.fit(X_trains, y_trains).score(trains_shift_cca[test_idx_all], classes[test_idx_all])

                scores[itera] = score.mean()
                # stds[itera] = score.std()
                itera -= 1
            
            results.append(['feat_TD4_cv_'+str(cv), 'lda_cca', n_components, 'S0 + '+channel_pos, np.mean(scores), np.std(scores)])
    
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
    print 'generate transform equations finished.........'


def training_lda_TD4_inter(my_clfs, trains_S0, trains_shift, classes, **kw):
    print 'training_lda_TD4_inter.........'
    start_time = time.time()

    log_fold = root_path + '/result/' + kw['log_fold']
    new_fold(log_fold)

    chan_len = kw['chan_len']
    action_num  = kw['action_num']

    print "----training "+kw['feature_type']+" inter, training by position O, testing by electrode shift "

    cv = 5
    results = []
    results.append(['Feat', 'Algorithm','Channel_Pos', 'Accuracy', 'std'])
    log_file = 'feat_'+kw['feature_type']+'_inter'

    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False,
                          tol=0.0001)

    data_num = trains_S0.shape[0]/action_num
    # print data_num
    
    scores = sklearn.cross_validation.cross_val_score(
        clf, trains_S0, classes, cv=cv)
    results.append(['feat_TD4_cv_'+str(cv), 'lda', 'S0',
                    scores.mean(), scores.std()])
    
    kf = KFold(data_num, n_folds=cv)
    
    for idx, channel_pos in enumerate(kw['pos_list']):

        X_test = trains_shift[:,idx*chan_len:idx*chan_len+chan_len]
        y_test = classes

        iteration = cv
        scores = np.zeros((iteration,))
        cca_scores = np.zeros((iteration,))
        
        
        iteration -= 1
        for train_idx, test_idx in kf:
            train_idx_all = np.array([], np.int)
            test_idx_all = np.array([], np.int)
            for action_idx in range(action_num):
                train_idx_all = np.concatenate( (train_idx_all, train_idx*action_idx), axis=0)
                test_idx_all = np.concatenate( (test_idx_all, test_idx*action_idx), axis=0)

            # X_train, y_train = trains_S0[train_idx_all], classes[train_idx_all]
            X_train, y_train = trains_S0, classes

            X_train_shift, y_train_shift = X_test[train_idx_all], classes[train_idx_all]
            
            X_train_all = np.concatenate( (X_train, X_train_shift), axis=0)
            y_train_all = np.concatenate( (y_train, y_train_shift), axis=0)
            
            sys.exit(0)
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