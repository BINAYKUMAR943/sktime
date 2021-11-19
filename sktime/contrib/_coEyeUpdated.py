# -*- coding: utf-8 -*-
"""Co-eye classifier.

Ensemble Classifier
"""
import warnings

warnings.filterwarnings("ignore")
from pyts.approximation import (
    SymbolicAggregateApproximation,
    SymbolicFourierApproximation,
)
from sktime.transformations.panel.dictionary_based import SFA
from sktime.transformations.panel.dictionary_based import SAX
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from pyts.preprocessing import StandardScaler
import random
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import LeaveOneOut
import heapq
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import os
from sktime.classification.base import BaseClassifier
from sklearn.utils.multiclass import class_distribution
from sklearn.preprocessing import KBinsDiscretizer
from numba import int64, prange, njit


class CoEye(BaseClassifier):
    """A Multi-resolution ensemble classifier for symbolically approximated time series.

    Implementation of Co-eye from Abdallah2020( 2020).[1]

    Overview: Input "n" series of length "m". Co-eye loops through different combination of
    of parameters(alpha and w) and calculates accuracy for each classifier using five fold cross
    validation. It retains all the classifiers which gives maximum accuracy or very
    close to it( with 1 percent margin) in the ensemble.

    _fit involves finding parameters pairs giving maximum accuracy or very close to it.

    Hyper-parameters:
    ------------------
        alpha: int
        alpha is the alphabet size
        w: int
        w is the word size in SAX and number of fourier coefficients in SFA

    Attributes
    ----------
    n_classes : int
        Number of classes. Extracted from the data.
    n_instances : int
        Number of instances. Extracted from the data.
    n_estimators : int
        The final number of classifiers used. Will be <= `max_ensemble_size` if
        `max_ensemble_size` has been specified.
    series_length : int
        Length of all series (assumed equal).
    classifiers : list
       List of RandomForest classifiers.
    class_dictionary: dict
        Dictionary of classes. Extracted from the data.


    References
    --------------
    [1]: Abdallah, Z.S., Gaber, M.M. Co-eye: a multi-resolution ensemble classifier
    for symbolically approximated time series. Mach Learn 109, 2029–2061 (2020).
    https://doi.org/10.1007/s10994-020-05887-3
    """

    _tags = {
        "coerce-X-to-numpy": True,
        "coerce-X-to-pandas": False,
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
    }

    def __init__(
        self,
        smote=False,
        maxBin=26,
        maxCoof=130,
        random_state=None,
    ):
        self.smote = smote
        self.maxCoof = maxCoof
        self.maxBin = maxBin
        self.random_state = random_state
        self.RFmatrices = []
        self.All_Acc = []
        self.SFA_acc = []
        self.SFA_pairs = []
        self.SAX_pairs = []
        self.classifiers = []
        self.n_classes = 0
        self.classes_ = []
        self._class_dictionary = {}

        super(CoEye, self).__init__()

    def _fit(self, X, y):
        """Fit time series classifier to training data.
        core logic
        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries
        y : array-like, shape = [n_instances] - the class labels
        Returns
        -------
        self : reference to self.
        State change
        ------------
        creates fitted model (attributes ending in "_")
        """

        X_1 = X
        # Changing three dimensional X to two dimension.
        X = X.reshape(X.shape[0], -1)

        min_neighbours = min(Counter(y).items(), key=lambda k: k[1])[1]
        max_neighbours = max(Counter(y).items(), key=lambda k: k[1])[1]
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        self.SMOTE_Xtrain = X
        self.SMOTE_ytrain = y

        # ========= Commenting to generate ouput without smote

        # if min_neighbours == max_neighbours:

        #     self.SMOTE_Xtrain = X
        #     self.SMOTE_ytrain = y

        #     # Apply SMOTE if data is imbalanced
        # else:
        #     if min_neighbours > 5:
        #         min_neighbours = 6
        #     self.SMOTE_Xtrain, self.SMOTE_ytrain = SMOTE(
        #         sampling_strategy="all", k_neighbors=min_neighbours - 1, random_state=42
        #     ).fit_resample(X, y)

        # ===================================================================

        # Training Phase
        # --------------

        # Store prediction probability for test data
        #   ======  Commenting to test SAX fast. Uncomment it after completion=========================
        # =================================needs to be deleted after successful run===========================================
        self.SFA_pairs = self._searchLense_SFA(
            self.SMOTE_Xtrain, self.SMOTE_ytrain, X_1
        )
        # ============================================================================================
        # self.SFA_pairs = self._searchLense_SFA(self.SMOTE_Xtrain, self.SMOTE_ytrain)
        for n_coefs, n_bins in self.SFA_pairs:
            SFA = SymbolicFourierApproximation(
                n_coefs=n_coefs, n_bins=n_bins, alphabet="ordinal"
            )
            # Transform to SFA
            X_train_SFA = SFA.fit_transform(self.SMOTE_Xtrain)
            # B Needs to be done in Predict function
            # X_test_SFA = SFA.fit_transform(X_test)
            # Build RF on each lense
            RF_clf = RandomForestClassifier(n_estimators=100, random_state=0)
            RF_clf.fit(X_train_SFA, self.SMOTE_ytrain)
            self.classifiers.append(RF_clf)

        # #===================================================================================================

        self.saxPairs = self._searchLense_SAX(self.SMOTE_Xtrain, self.SMOTE_ytrain, X_1)
        for n_bins in self.saxPairs:
            sax = SymbolicAggregateApproximation(
                strategy="uniform", n_bins=n_bins, alphabet="ordinal"
            )
            # Transform to SAX
            # SMOTE_Xtrain=SMOTE_Xtrain.reshape(SMOTE_Xtrain.shape[0], -1)
            X_sax = sax.fit_transform(self.SMOTE_Xtrain)
            # X_test_sax = sax.fit_transform(X_test)
            # Build RF for each SAX lense
            RF_clf = RandomForestClassifier(n_estimators=100, random_state=0)
            RF_clf.fit(X_sax, self.SMOTE_ytrain)
            self.classifiers.append(RF_clf)
        self.n_estimators = len(self.classifiers)
        # Store prediction probability for test data
        # B Needs to be done in pred
        # model_pred= RF_clf.predict_proba(X_test_SFA)
        # accumulate RFmatrices for a lense
        # RFmatrices.append(model_pred)

        return self

        # Transform to SAX Lenses

        # Classification Phase

    def _predict(self, X):
        """Predict labels for sequences in X.
        core logic
        Parameters
        ----------
        X : 3D np.array, array-like or sparse matrix
                of shape = [n_instances,n_dimensions,series_length]
                or shape = [n_instances,series_length]
            or single-column pd.DataFrame with pd.Series entries
        Returns
        -------
        y : array-like, shape =  [n_instances] - predicted class labels
        """
        X_1 = X
        X = X.reshape(X.shape[0], -1)
        clf_nbr = 0

        for n_coefs, n_bins in self.SFA_pairs:
            # ===========================Commenting to try with sktime dependencies==============================
            # SFA = SymbolicFourierApproximation(
            #     n_coefs=n_coefs, n_bins=n_bins, alphabet="ordinal"
            # )
            # # transform to SFA
            # X_test_SFA = SFA.fit_transform(X)

            # ==========================================================================================================
            sfa = SFA(
                word_length=n_coefs,
                alphabet_size=n_bins,
                binning_method="equi-depth",
                save_words=True,
                window_size=X.shape[1],
            )
            X_SFA_fit = sfa.fit(X_1)
            X_SFA_transform = X_SFA_fit.transform(X_1)
            # print(sfa.word_list(541190))
            sfa_list = []
            for i in X_SFA_transform[0]:
                x = list(i.keys())
                sfa_list.append(sfa.word_list(x[0]))

            # print("Printing sktime SFA transormed")
            X_test_SFA = np.array(list(sfa_list))
            # ========================================================================================================

            RF_clf = self.classifiers[clf_nbr]
            model_pred = RF_clf.predict_proba(X_test_SFA)
            # self.n_classes=RF_clf.classes_
            # accumulate RFmatrices for a lense
            self.RFmatrices.append(model_pred)
            clf_nbr = clf_nbr + 1

        for n_bins in self.saxPairs:
            # =====================Commenting to try with sax dependencies===============================================================
            # sax = SymbolicAggregateApproximation(
            #     strategy="uniform", n_bins=n_bins, alphabet="ordinal"
            # )
            # # Transform to SAX
            # X_test_sax = sax.fit_transform(X)
            # =============================================================================================================================
            kb = KBinsDiscretizer(n_bins=n_bins, strategy="uniform", encode="ordinal")
            # kb1=self._kbdisctretizer(X_train,alpha)
            sax = kb.fit_transform(X)
            X_test_sax = sax.astype(int)
            # ==================================================================================================================================================
            RF_clf = self.classifiers[clf_nbr]
            model_pred = RF_clf.predict_proba(X_test_sax)
            # accumulate RFmatrices for all lenses
            self.RFmatrices.append(model_pred)
            clf_nbr = clf_nbr + 1

        SFA_pairs_length = len(self.SFA_pairs)
        y_pred = self._mostConf_multi(
            self.RFmatrices, RF_clf.classes_, SFA_pairs_length
        )
        return y_pred

    # Find the most confident label across lenses in one representation (either SAX or SFA)
    # def _mostConf_one(self, matrcies, labels):
    #     # Marerices is of length L: number of lenses/forests
    #     # Each matrix in matrices is the probalistic accuracy of test data
    #     # Matrix of size m X n: where m is the number of test instances,n is the number of classes
    #     predLabels = []
    #     # For each test data
    #     for row in range(len(matrcies[0])):
    #         maxConf = 0
    #         # For each Forest/lense
    #         for mat in matrcies:
    #             for col in range(len(labels)):
    #                 if mat[row][col] > maxConf:
    #                     maxConf = mat[row][col]
    #                     Conflabel = labels[col]
    #         # Return the most confident lable across lenses
    #         predLabels.append(Conflabel)
    #     return predLabels

    # ### Implementation of Vote function

    # #Find the most confident labeles across lenses of the two representations ( SAX and SFA)

    def _mostConf_multi(self, matrcies, labels, sfa_n):
        # Matrices is of length L: number of lenses/forests (total for SAX and SFA)
        # Each matrix in matrices is the probalistic accuracy of test data
        # Matrix of size m X n: where m is the number of test instances,n is the number of classes (labels)
        # Sax_n number of lenses in SFA

        predLabel = []
        multiSFA = False
        multiSAX = False

        # B For each test data, it checks every matrix and finds the most confident one and its corresponding label.
        # B So for for each test data it run for all matrix and check all the possible labels( for binary there are two lables)
        # B  and save the maximum value as most confident one and also save its label.

        # For each test data, we define the most confident and second-most confident for each presentation, then vote between them
        for row in range(len(matrcies[0])):
            # Initialise
            maxConf_SFA = 0
            # B SFAConf is for storing all the probabilities value in a list
            SFAConf = []
            # SFA Labels is for stroing all the labels in a list
            SFALables = []
            #
            SAXConf = []
            SAXLables = []
            # Look into SFA only matrices
            # Find the max confidence/ prbability in SFA
            for mat in matrcies[:sfa_n]:
                for col in range(len(labels)):
                    SFAConf.append(mat[row][col])
                    SFALables.append(labels[col])
                    if mat[row][col] > maxConf_SFA:
                        maxConf_SFA = mat[row][col]
                        Conflabel_SFA = labels[col]

            # Second best flag
            SB1 = False
            # Special case: when multiple lenses have the same accuracy (most confident)
            if SFAConf.count(maxConf_SFA) > 1:
                indices = [i for i, x in enumerate(SFAConf) if x == maxConf_SFA]
                l = [SFALables[i] for i in indices]
                if len(set(l)) != 1:
                    maxIr = max(l.count(y) for y in set(l))
                    k = [l.count(y) for y in set(l)]

                    # Find the most common label
                    if k.count(maxIr) == 1:
                        cnt = Counter(l)
                        Conflabel_SFA = cnt.most_common(2)[0][0]
                        secondBestSFALabel = cnt.most_common(2)[1][0]

                    # If no common label, in case of tie, the label is chosen randomly.
                    else:
                        shuff = []
                        for it in set(l):
                            if l.count(it) == maxIr:
                                shuff.append(it)
                        shuff = list(set(l))
                        random.shuffle(shuff)
                        Conflabel_SFA = shuff[0]
                        secondBestSFALabel = shuff[1]
                    SB1 = True
                    # Set the second best
                    secondBestSFA = maxConf_SFA

            if SB1 == False:
                secondBestSFA = max(n for n in SFAConf if n != maxConf_SFA)
                secondBestSFALabel = SFALables[SFAConf.index(secondBestSFA)]

            # Same steps for SAX
            maxConf_SAX = 0
            SB2 = False
            for mat in matrcies[sfa_n:]:
                for col in range(len(labels)):
                    SAXConf.append(mat[row][col])
                    SAXLables.append(labels[col])
                    if mat[row][col] > maxConf_SAX:
                        maxConf_SAX = mat[row][col]
                        Conflabel_SAX = labels[col]
            if SAXConf.count(maxConf_SAX) > 1:
                indices = [i for i, x in enumerate(SAXConf) if x == maxConf_SAX]
                l = [SAXLables[i] for i in indices]
                if len(set(l)) != 1:
                    print("Conflict on max confident", l)
                    maxIr = max(l.count(y) for y in set(l))
                    k = [l.count(y) for y in set(l)]
                    #                 How many equal items with max confident value
                    if k.count(maxIr) == 1:
                        cnt = Counter(l)
                        Conflabel_SAX = cnt.most_common(2)[0][0]
                        secondBestSAXLabel = cnt.most_common(2)[1][0]
                    else:
                        shuff = []
                        for it in set(l):
                            if l.count(it) == maxIr:
                                shuff.append(it)
                        print("tie", shuff)
                        random.shuffle(shuff)
                        Conflabel_SAX = shuff[0]
                        secondBestSAXLabel = shuff[1]

                    secondBestSAX = maxConf_SAX
                    #                 print("most common", Conflabel_SAX, "Second Best", secondBestSAXLabel)
                    SB2 = True
            #             print("indecies", indices, "labels" ,labels)
            if SB2 == False:
                secondBestSAX = max(n for n in SAXConf if n != maxConf_SAX)
                secondBestSAXLabel = SAXLables[SAXConf.index(secondBestSAX)]

            #         print ("Best SAX",maxConf_SAX , Conflabel_SAX)
            #         print ("Second Best SAX",secondBestSAX , secondBestSAXLabel)

            # -----------------------------------
            # In case of agreement between most Conf SAX and SFA
            if Conflabel_SAX == Conflabel_SFA:
                best = Conflabel_SAX
            # If no agreement, then second best is testes
            elif secondBestSAX > secondBestSFA:
                best = secondBestSAXLabel
            else:
                best = secondBestSFALabel
            # Accumulate labels with the best choice
            predLabel.append(best)
        #         print("Best SFA, SAX", Conflabel_SAX, Conflabel_SFA,"Second best SFA, SAX", secondBestSAXLabel, secondBestSFALabel,"Best",  best)
        #         print ("----------------------------------------")

        return predLabel

    # ### searchLenses [Algorithm 2 (Co-eye paper)]

    # Finding best pairs for SFA transformation
    def _searchLense_SFA(self, X_train, y_train, X_1):
        # Input is training data (X_train, y_train)
        # Returns selected pairs for SFA transformation based on cross validation
        ram = 0

        # Set ranges (Seg, alpha) parameters
        print("\n Generating SFA parameters....")
        # maxCoof = 130
        if X_train.shape[1] < self.maxCoof:
            self.maxCoof = X_train.shape[1] - 1
        if X_train.shape[1] < 100:
            n_segments = list(range(5, self.maxCoof, 5))
        else:
            n_segments = list(range(10, self.maxCoof, 10))

        self.maxBin = 26
        if X_train.shape[1] < self.maxBin:
            self.maxBin = X_train.shape[1] - 2
        if X_train.shape[0] < self.maxBin:
            self.maxBin = X_train.shape[0] - 2
        alphas = range(3, self.maxBin)

        pairs = []

        # Learning parameteres using 5 folds cross validation

        for alpha in alphas:

            s = []
            for seg in n_segments:

                # ==============================Commenting for recreating result in sktime===============================================================
                sfa = SFA(
                    word_length=seg,
                    alphabet_size=alpha,
                    binning_method="equi-depth",
                    save_words=True,
                    window_size=X_train.shape[1],
                )
                # +============================================================For pyts package===========================================================================================
                #     sfa_pyts=SymbolicFourierApproximation(
                #     n_coefs=seg, n_bins=alpha, alphabet='ordinal',drop_sum=True
                # )
                #     X_sfa=sfa_pyts.fit_transform(X_train)
                #     if ram==0:
                #         print("Length of time series = "+ str(X_train.shape[1]))
                #         print('=======================================')
                #         print("Printing first time series SFA transform using pyts package")
                #         print(X_sfa[0])
                #         print('========================================')
                #         ram=ram+1
                # #+============================== Converting SFA outut into a numpy array=====================================================
                X_SFA_fit = sfa.fit(X_1)
                X_SFA_transform = X_SFA_fit.transform(X_1)
                # print(sfa.word_list(541190))
                sfa_list = []
                for i in X_SFA_transform[0]:
                    x = list(i.keys())
                    sfa_list.append(sfa.word_list(x[0]))

                # print("Printing sktime SFA transormed")
                sfa_array = np.array(list(sfa_list))
                # print(" Printing pyts SFA transformed")
                # if ram==1:
                #     print("Length of time series = "+ str(X_train.shape[1]))
                #     print('=======================================')
                #     print("Printing first time series SFA transform using sktime sfa package")
                #     print(sfa_array[0])
                #     print('========================================')
                #     ram=ram+1
                # #==============================================================================================
                # print("bag of string"+sfa.bag_to_string(X_SFA_transform[0][0]))

                # word=sfa.words
                # print"Printing words",X_SFA_transform)
                # print()
                # print( type(X_SFA_transform),"type of returned data")
                # print(X_SFA)
                # sfa_sktime=X_SFA_transform[0]
                # ================= With sfa() =========================================

                # print((X_mSFA_))
                # #===========
                # print(type(X_mSFA_))
                scores = 0
                RF_clf = RandomForestClassifier(n_estimators=100, random_state=0)
                scores = cross_val_score(RF_clf, sfa_array, y_train, cv=5)
                s.append(scores.mean())

            winner = np.argwhere(s >= np.amax(s) - 0.01)

            for i in winner.flatten().tolist():
                bestCof = n_segments[i]
                pairs.append((bestCof, alpha))
        print("No of selected pairs: ", len(pairs))
        print("SFA pamaeter selection, done!")
        return pairs

    # Finding best pairs for SAX transformation

    def _searchLense_SAX(self, X_train, y_train, X_1):
        # Input is training data (X_train, y_train)
        # Returns selected pairs for SAX transformation based on cross validation
        # X_train = X_train.reshape(X_train.shape[0], -1)
        # lakshman=0
        # Set range (alpha) parameter
        print("Generating SAX parameters....")

        # self.maxBin = 26
        if X_train.shape[1] < self.maxBin:
            self.maxBin = X_train.shape[1]
        alphas = range(3, self.maxBin)
        s = []
        pairs = []

        # Learning parameters using 5 folds cross validation
        for alpha in alphas:

            ###########==================Commenting for using just kbins discretizer==================================================
            # SAX_pyts = SymbolicAggregateApproximation(
            #     strategy="uniform", n_bins=alpha, alphabet="ordinal"
            # )
            # X_train_SAX = SAX_pyts.fit_transform(X_train)
            # if lakshman==0:
            #     print("Length of time series = "+ str(X_train.shape[1]))
            #     print('Printing first time series sax transform using pyts')
            #     print('=======================================')
            #     print(X_train_SAX[0])
            #     print('========================================')
            #     lakshman=lakshman+1
            # =====================================================================================
            # ===========Commenting to run the algorithm without smote=======================================
            # sax_sktime=SAX(window_size=X_train.shape[1],alphabet_size= alpha,save_words=True , word_length=X_train.shape[1])
            # X_train_sax_sktime=sax_sktime.fit_transform(X_1)
            # print(sax_sktime.words)
            # ================================================================================================
            # print(X_train_sax_sktime)
            # print(X_train_sax_sktime)

            # print(X_train_SAX[0])

            # ============================Trying using KBInDiscretizer=====================================================
            kb = KBinsDiscretizer(n_bins=alpha, strategy="uniform", encode="ordinal")
            # kb1=self._kbdisctretizer(X_train,alpha)

            sax = kb.fit_transform(X_train)
            sax = sax.astype(int)
            # if lakshman==1:
            #     print("Length of time series = "+ str(X_train.shape[1]))
            #     print('Printing first time series sax transform using KBinsdiscretizer')
            #     print('=======================================')
            #     print(sax[0])
            #     print('========================================')
            #     lakshman=lakshman+1
            # print(sax)

            # ==============================================================================================================

            scores = 0
            RF_clf = RandomForestClassifier(n_estimators=100, random_state=0)
            scores = cross_val_score(RF_clf, sax, y_train, cv=5)
            s.append(scores.mean())
        winner = np.argwhere(s >= np.mean(s) - 0.01)
        for i in winner.flatten().tolist():
            bestCof = alphas[i]
            pairs.append(bestCof)

        print("No of selected pairs: ", len(pairs))
        print("SAX pamaeter selection, done!")

        return pairs

    def _predict_proba(self, X):
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_instances, 1)

        Returns
        -------
        dists : array of shape (n_instances, n_classes)
            Predicted probability of each class.
        """
        preds = self._predict(X)
        print(X.shape[0])
        print(type(X))
        print(self.n_classes)

        dists = np.zeros((X.shape[0], self.n_classes))

        for i in range(0, X.shape[0]):
            dists[i, self._class_dictionary.get(preds[i])] += 1

        return dists

        # =========================Implementing methods for KBinDiscretizer==========================================

    def _kbdisctretizer(self, X, n_bins):
        n_samples, n_timestamps = X.shape
        sample_min, sample_max = np.min(X, axis=1), np.max(X, axis=1)
        # bin_edges = np.empty((n_bins - 1, n_samples))

        bin_edges = _uniform_bins(sample_min, sample_max, n_samples, n_bins)
        X_new = _digitize_2d(X, bin_edges, n_samples, n_timestamps)
        print(X_new)


@njit()
def _uniform_bins(sample_min, sample_max, n_samples, n_bins):
    bin_edges = np.empty((n_bins - 1, n_samples))
    for i in prange(n_samples):
        bin_edges[:, i] = np.linspace(sample_min[i], sample_max[i], n_bins + 1)[1:-1]
    return bin_edges


@njit()
def _digitize_2d(X, bins, n_samples, n_timestamps):
    X_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_samples):
        X_digit[i] = np.digitize(X[i], bins[i], right=True)
    return X_digit
