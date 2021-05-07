# -*- coding: utf-8 -*-
# import packages
import time
from tqdm import tqdm
import warnings
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.metrics import val_model_binary

warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
from collections import Counter

class SelfPacedSample():
    def __init__(self,
                 hardness_func=lambda y_true, y_pred: np.absolute(y_true - y_pred),
                 n_estimators=10,
                 k_bins=10,
                 estimator_params=tuple(),
                 n_jobs=None,
                 random_state=None,
                 verbose=0, ):

        self.hardness_func = hardness_func
        self.n_estimators = n_estimators
        self.k_bins = k_bins
        self.estimator_params = estimator_params
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _random_under_sampling(self, X_maj, y_maj, X_min, y_min):
        """Private function used to perform random under-sampling."""

        np.random.seed(self.random_state)
        idx = np.random.choice(len(X_maj), len(X_min), replace=False)
        X_train = np.concatenate([X_maj[idx], X_min])
        y_train = np.concatenate([y_maj[idx], y_min])

        return X_train, y_train

    def _self_paced_under_sampling(self, X_maj, y_maj, X_min, y_min, hardness, i_estimator):
        step = (hardness.max() - hardness.min()) / self.k_bins
        bins = []
        ave_contributions = []
        for i_bins in range(self.k_bins):
            idx = (
                    (hardness >= i_bins * step + hardness.min()) &
                    (hardness < (i_bins + 1) * step + hardness.min())
            )
            # Marginal samples with highest hardness value -> kth bin
            if i_bins == (self.k_bins - 1):
                idx = idx | (hardness == hardness.max())
            bins.append(X_maj[idx])
            ave_contributions.append(hardness[idx].mean())

        # Update self-paced factor alpha
        alpha = np.tan(np.pi * 0.5 * (i_estimator / (self.n_estimators - 1)))
        # Caculate sampling weight
        weights = 1 / (ave_contributions + alpha)
        weights[np.isnan(weights)] = 0
        # Caculate sample number from each bin
        n_sample_bins = len(X_min) * weights / weights.sum()
        n_sample_bins = n_sample_bins.astype(int) + 1

        # Perform self-paced under-sampling
        sampled_bins = []
        for i_bins in range(self.k_bins):
            if min(len(bins[i_bins]), n_sample_bins[i_bins]) > 0:
                np.random.seed(self.random_state)
                idx = np.random.choice(
                    len(bins[i_bins]),
                    min(len(bins[i_bins]), n_sample_bins[i_bins]),
                    replace=False)
                sampled_bins.append(bins[i_bins][idx])
        X_train_maj = np.concatenate(sampled_bins, axis=0)
        y_train_maj = np.full(X_train_maj.shape[0], y_maj[0])

        # Handle sparse matrix
        if sp.issparse(X_min):
            X_train = sp.vstack([sp.csr_matrix(X_train_maj), X_min])
        else:
            X_train = np.vstack([X_train_maj, X_min])
        y_train = np.hstack([y_train_maj, y_min])

        return X_train, y_train

    def _self_under_sampling(self, Xs_maj, ys_maj, Xs_min, ys_min, Xtl_maj, ytl_maj, Xtl_min, ytl_min, i_estimator):
        """Private function used to perform self-paced under-sampling."""

        # Update hardness value estimation
        hardness_s = self.hardness_func(ys_maj, self.ys_maj_pred_proba_buffer[:, self.class_index_min_s])
        hardness_t = self.hardness_func(ytl_maj, self.ytl_maj_pred_proba_buffer[:, self.class_index_min_t])

        # If hardness values are not distinguishable, perform random smapling
        if hardness_s.max() == hardness_s.min():
            Xs_train, ys_train = self._random_under_sampling(Xs_maj, ys_maj, Xs_min, ys_min)
        else:
            Xs_train, ys_train = self._self_paced_under_sampling(Xs_maj, ys_maj, Xs_min, ys_min, hardness_s, i_estimator)

        if hardness_t.max() == hardness_t.min():
            Xtl_train, ytl_train = self._random_under_sampling(Xtl_maj, ytl_maj, Xtl_min, ytl_min)
        else:
            Xtl_train, ytl_train = self._self_paced_under_sampling(Xtl_maj, ytl_maj, Xtl_min, ytl_min, hardness_t, i_estimator)

        return Xs_train, ys_train, Xtl_train, ytl_train


    def update_maj_pred_buffer(self, model, Xs_maj, Xtl_maj):
        """Maintain a latest prediction probabilities of the majority
           training data during ensemble training."""


        if self.n_buffered_estimators_ == 0:
            self.ys_maj_pred_proba_buffer = np.full(shape=(self._n_samples_maj_s, self.n_classes_),
                                                   fill_value=1. / self.n_classes_)

            self.ytl_maj_pred_proba_buffer = np.full(shape=(self._n_samples_maj_t, self.n_classes_),
                                                   fill_value=1. / self.n_classes_)

        ys_maj_pred_proba_buffer = self.ys_maj_pred_proba_buffer
        ytl_maj_pred_proba_buffer = self.ytl_maj_pred_proba_buffer

        for i in range(self.n_buffered_estimators_, len(self.probs_test)):
            ys_pred_proba_i = model.predict(Xs_maj)[0]
            ys_maj_pred_proba_buffer = (ys_maj_pred_proba_buffer * i + ys_pred_proba_i) / (i + 1)

            ytl_pred_proba_i = model.predict(Xtl_maj)[0]
            ytl_maj_pred_proba_buffer = (ytl_maj_pred_proba_buffer * i + ytl_pred_proba_i) / (i + 1)

        self.ys_maj_pred_proba_buffer = ys_maj_pred_proba_buffer
        self.ytl_maj_pred_proba_buffer = ytl_maj_pred_proba_buffer
        self.n_buffered_estimators_ = len(self.probs_test)

        return

    def init_data_statistics(self, Xs, ys, Xtl, ytl, label_maj, label_min, to_console=False):
        """Initialize DupleBalance with training data statistics."""

        self._n_samples_s, self.n_features_s = Xs.shape
        self.features_s = np.arange(self.n_features_s)
        self.org_class_distr_s = Counter(ys) # 记录出现次数

        self._n_samples_t, self.n_features_t = Xtl.shape
        self.features_t = np.arange(self.n_features_t)
        self.org_class_distr_t = Counter(ytl) # 记录出现次数

        self.classes_ = np.unique(ys)
        self.n_classes_ = len(self.classes_)
        self.n_buffered_estimators_ = 0

        if self.n_classes_ != 2:
            raise ValueError(f"Number of classes should be 2, meet {self.n_classes_}, please check usage.")

        if label_maj == None or label_min == None:
            # auto detect majority and minority class label
            sorted_class_distr_s = sorted(self.org_class_distr_s.items(), key=lambda d: d[1])
            label_min, label_maj = sorted_class_distr_s[0][0], sorted_class_distr_s[1][0]
            if to_console:
                print(f'\n\'label_maj\' and \'label_min\' are not specified, automatically set to {label_maj} and {label_min}')

        self.label_maj, self.label_min = label_maj, label_min
        self.class_index_maj_s, self.class_index_min_s = list(self.classes_).index(label_maj), list(self.classes_).index(label_min)
        self.class_index_maj_t, self.class_index_min_t = list(self.classes_).index(label_maj), list(self.classes_).index(label_min)

        maj_index_s, min_index_s = (ys == label_maj), (ys == label_min)
        self._n_samples_maj_s, self._n_samples_min_s = maj_index_s.sum(), min_index_s.sum()

        maj_index_t, min_index_t = (ytl == label_maj), (ytl == label_min)
        self._n_samples_maj_t, self._n_samples_min_t = maj_index_t.sum(), min_index_t.sum()

        if self._n_samples_maj_s == 0 | self._n_samples_maj_t == 0:
            raise RuntimeWarning(
                f'The specified majority class {self.label_maj} has no data samples, please check usage.')
        if self._n_samples_min_s == 0 | self._n_samples_min_t == 0 :
            raise RuntimeWarning(
                f'The specified minority class {self.label_min} has no data samples, please check usage.')

        self.Xs_maj, self.ys_maj = Xs[maj_index_s], ys[maj_index_s]
        self.Xs_min, self.ys_min = Xs[min_index_s], ys[min_index_s]

        self.Xtl_maj, self.ytl_maj = Xtl[maj_index_t], ytl[maj_index_t]
        self.Xtl_min, self.ytl_min = Xtl[min_index_t], ytl[min_index_t]

        if to_console:
            print('-----------------Source Domain----------------------')
            print('# Samples       : {}'.format(self._n_samples_s))
            print('# Features      : {}'.format(self.n_features_s))
            print('# Classes       : {}'.format(self.n_classes_))
            cls_label, cls_dis, IRs = '', '', ''
            min_n_samples_s = min(self.org_class_distr_s.values())
            for label, num in sorted(self.org_class_distr_s.items(), key=lambda d: d[1], reverse=True):
                cls_label += f'{label}/'
                cls_dis += f'{num}/'
                IRs += '{:.2f}/'.format(num / min_n_samples_s)
            print('Classes         : {}'.format(cls_label[:-1]))
            print('Class Dist      : {}'.format(cls_dis[:-1]))
            print('Imbalance Ratio : {}'.format(IRs[:-1]))
            print('----------------------------------------------------')
            time.sleep(0.25)

            print('\n-----------------Target Domain----------------------')
            print('# Samples       : {}'.format(self._n_samples_t))
            print('# Features      : {}'.format(self.n_features_t))
            print('# Classes       : {}'.format(self.n_classes_))
            cls_label, cls_dis, IRs = '', '', ''
            min_n_samples_t = min(self.org_class_distr_t.values())
            for label, num in sorted(self.org_class_distr_t.items(), key=lambda d: d[1], reverse=True):
                cls_label += f'{label}/'
                cls_dis += f'{num}/'
                IRs += '{:.2f}/'.format(num / min_n_samples_t)
            print('Classes         : {}'.format(cls_label[:-1]))
            print('Class Dist      : {}'.format(cls_dis[:-1]))
            print('Imbalance Ratio : {}'.format(IRs[:-1]))
            print('----------------------------------------------------')
            time.sleep(0.25)

        return


    def fit(self, model, Xs, ys, Xtl, ytl, Xtu, Xtest, ytest,label_maj=None, label_min=None):
        # Initialize by spliting majority / minority set
        self.init_data_statistics(
            Xs, ys, Xtl, ytl, label_maj, label_min,
            to_console=True if self.verbose > 0 else False)

        self.probs_test = []
        # self.probs_s = []
        # self.probs_tl = []

        # Loop start
        if self.verbose > 0:
            iterations = tqdm(range(self.n_estimators))
            iterations.set_description('SPE Training')
        else:
            iterations = range(self.n_estimators)

        for i_iter in iterations:
            print(f"\n========SPE iteration {i_iter} ========")
            # update current majority training data prediction
            self.update_maj_pred_buffer(model, self.Xs_maj, self.Xtl_maj)

            # train a new base estimator and add it into self.estimators_
            Xs_train, ys_train, Xtl_train, ytl_train = self._self_under_sampling(self.Xs_maj, self.ys_maj, self.Xs_min, self.ys_min,
                                                                                 self.Xtl_maj, self.ytl_maj, self.Xtl_min, self.ytl_min,i_iter)

            print(f"Xs_train: {Xs_train.shape}\tXtl_train: {Xtl_train.shape}\n")

            ys_train_cat = to_categorical(ys_train)
            ytl_train_cat = to_categorical(ytl_train)
            ytest_cat = to_categorical(ytest)

            model.fit(Xs_train, ys_train_cat, Xtl_train, ytl_train_cat, Xtu, Xtest, ytest_cat, target_label=None, n_iter=1000, cal_bal=False)
            prob_t = model.evaluate(Xtest)

            val_model_binary(ytest, prob_t)

            self.probs_test.append(prob_t)

        return self
