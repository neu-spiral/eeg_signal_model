import numpy as np
import warnings
from copy import copy
from sklearn import metrics, linear_model
import pyprog

eps = np.power(.1, 7)
CRED = '\033[31m'
CEND = '\033[0m'

warnings.filterwarnings("ignore")


class ARXModel(object):
    """
        Given the EEG data and visual stimuli information across sequences,
        this class fits an ARX model to EEG signals.
        For more information, check the following papers:

        - Y. M. Marghi, et al., "A Parametric EEG Signal Model for BCIs with
          Rapid-Trial Sequences", EMBC,2018.
        - Y. M. Marghi, et al., "An Event-Driven AR-Process Model With
          RapidTrial Sequences for EEG-based BCIs", TNSRE,2019.

        Attributes:
            orderSelection ---> Boolean
            paradigm ---> the experiment paradigm, either FRP or ERP, char
            compOrder ---> polynomial order of the gamma function, int
            threshold ---> convergence criteria in cyclyc decent method, double
            channels ---> list of EEG channels or sources, char
            num_trial ---> number of trial in a sequence, int
            num_sample ---> number of data sample in a sequence, int
            order_AR ---> AR model order, could be a vector or scaler, int
            delays --->  delays of gamma functions, vector, double
            num_seq ---> total number of sequences, int
            tau --->  Kurtosis of gamma functions, vector, int
            fs ---> sampling frequency (Hz), scaler real value, double

        Methods:
            data_for_corssValidation
            cyclic_decent_method
            multiChanenelCoeff
            syntheticEEGseq
            gammafunction
            loglikelihood
            calculateROC
            calculateACC
            model_eval
            arburg_
            arburg
            invCov
    """

    def __init__(self, fs, paradigm, num_trial, num_seq, num_sample,
                 hyperparameter, channels, threshold=1e-6,
                 orderSelection=False):
        self.fs = fs
        self.paradigm = paradigm
        self.channels = channels
        self.num_sample = num_sample
        self.numSeq = num_seq
        self.numTrial = num_trial

        if orderSelection:
            self.order_AR = hyperparameter[:, 0]
            self.tau = hyperparameter[:, 1:4]
            self.delays = hyperparameter[:, 4:7]
            self.compOrder = hyperparameter[0, 7:]
        else:
            self.order_AR = hyperparameter[0]
            self.tau = hyperparameter[1:4]
            self.delays = hyperparameter[4:7]
            self.compOrder = hyperparameter[7:]

        self.threshold = threshold
        self.orderSelection = orderSelection

    def ARXmodelfit(self, data, nFold=10):
        """
            fits an ARX model to the multi-channel/ brain sources timeseries
            signal using the cyclic decent method
            Input Args:
                data: a dictionary including timeseries, onsets of stimuli,
                        and targets for all sequences
                nFold: number of folds(int)
                orderSelection: True if you want to set hyperparmeters by BIC
                                and grid search (boolean)
            Return:
                auc: auc of the classifier at each fold (nFold x 1)
                acc: accuracy of the classifier at each fold (nFold x 1)
                parameters: best parameters for each channel/brain sources
                hyperParam: hyperparameters of the ARX model for all channel
        """
        # Initialization
        auc_ch = np.zeros((nFold, len(self.channels)))
        auc = np.zeros((nFold, 1))
        acc = np.zeros((nFold, 1))
        # the maximum size of the parameters assumed to be 300
        parameter_hat = np.zeros((nFold, 300, len(self.channels)))
        trialTargetness = []
        score = []
        count = 0

        # Time resolution for delays in grid search for model order selection
        if self.orderSelection:
            dD = np.round(.005 * float(self.fs))
            delay0 = [int(self.delays[0][0] + i * dD) for i in
                      range(int(
                          (self.delays[1][0] - self.delays[0][0]) * dD + 1))]
            delay1 = [int(self.delays[0][1] + i * dD) for i in
                      range(int(
                          (self.delays[1][1] - self.delays[0][1]) * dD + 1))]
            delay2 = [int(self.delays[0][2] + i * dD) for i in
                      range(int(
                          (self.delays[1][2] - self.delays[0][2]) * dD + 1))]
            AR_range = range(self.order_AR[0], self.order_AR[1] + 1)
            tau0_range = range(self.tau[0][0], self.tau[1][0] + 1)
            tau1_range = range(self.tau[0][1], self.tau[1][1] + 1)
            tau2_range = range(self.tau[0][2], self.tau[1][2] + 1)

        # training the model within K-fold cross validation
        y_test, y_train, us_test, us_train, ue_test, ue_train = \
            self.data_for_corssValidation(data, nFold)

        # Create Object
        pb = pyprog.ProgressBar(" ", "", total=nFold * len(self.channels))
        # Update Progress Bar
        pb.update()
        for f in range(nFold):
            data_train = dict()
            data_test = dict()
            data_train.update(
                {"timeseries": y_train[f], "stimOnset": us_train[f],
                 "targetOnset": ue_train[f], "numSeq": ue_train[f].shape[1]})
            data_test.update({"timeseries": y_test[f], "stimOnset": us_test[f],
                              "targetOnset": ue_test[f],
                              "numSeq": ue_test[f].shape[1]})
            # Parameters estimation for each channel/brain sources
            numSeq_train = data_train["numSeq"]
            numSeq_test = data_test["numSeq"]
            for ch in range(len(self.channels)):
                if self.orderSelection:
                    bic = []
                    for k in AR_range:
                        hyperparameters = []
                        error = []
                        auc = []
                        for tau0 in [tau0_range[0]]:
                            for tau1 in [tau1_range[0]]:
                                for tau2 in [tau2_range[0]]:
                                    for d0 in [delay0[0]]:
                                        for d1 in [delay1[0]]:
                                            for d2 in [delay2[0]]:
                                                self.order_AR = k
                                                self.tau = [tau0, tau1, tau2]
                                                self.delays = [d0, d1, d2]
                                                self.numSeq
                                                param, _, _, _, _, _ = \
                                                    self.cyclic_decent_method(
                                                        data_train, ch)
                                                data_test["coeff"] = 1
                                                auc_, _, _, _ = self.model_eval(
                                                    param,
                                                    data_test, [ch])
                                                _, loglike, sigma_hat, \
                                                _, _, _ = \
                                                    self.cyclic_decent_method(
                                                        data_test, ch)
                                                hyperparameters.append([k,
                                                                        self.tau,
                                                                        self.delays,
                                                                        self.compOrder])
                                                error.append(sigma_hat[0])
                                                auc.append(auc_)
                        # Find the optimal ARorder, tau, delay, and compOrder according to auc
                        indx1 = [i for i in range(len(auc)) if
                                 auc[i] == max(auc)]
                        indx2 = [i for i in range(len(indx1)) if
                                 error[i] == min(error)]

                        if indx1 and indx2:
                            self.tau = hyperparameters[indx1[indx2[0]]][1]
                            self.delays = hyperparameters[indx1[indx2[0]]][2]
                            self.compOrder = hyperparameters[indx1[indx2[0]]][3]
                            hyperparameters = []
                            hyperparameters.append([k, self.tau, self.delays,
                                                    self.compOrder])

                        _, L, _, _, _, _ = self.cyclic_decent_method(data_test,
                                                                     ch)
                        # Compute BIC measure for AR model order selection
                        bic.append(
                            L + k * np.log(numSeq_train * (numSeq_test - k)))

                    indx = [i for i in range(len(bic)) if bic[i] == min(bic)]
                    self.order_AR = AR_range[indx[0]]
                    self.tau = hyperparameters[indx[0]][1]
                    self.delays = hyperparameters[indx[0]][2]
                    self.compOrder = hyperparameters[indx[0]][3]

                nParam = [self.order_AR, self.order_AR + sum(self.compOrder),
                          self.order_AR + sum(self.compOrder) + 1]
                param, _, _, _, _, _ = \
                    self.cyclic_decent_method(data_train, ch)
                if self.paradigm == "FRP":
                    parameter_hat[f, 0:2 * nParam[-1], ch] = param[:, 0]
                else:
                    parameter_hat[f, 0:nParam[-1], ch] = param[:, 0]

                auc_ch[f, ch], acc_ch, score_, trialTargetness = \
                    self.model_eval(param, data_test, [ch])
                score.append(score_)

            scores = np.zeros((score[0].shape[0], len(self.channels)))

            for ch in range(len(self.channels)):
                scores[:, ch] = score[ch]

            _, _, data_test["coeff"], _ = self.multiChanenelCoeff(scores,
                                                                  trialTargetness)
            if self.paradigm == "FRP":
                auc[f], acc[f], _, _ = self.model_eval(
                    parameter_hat[f, 0:2 * nParam[-1], :],
                    data_test, range(len(self.channels)))
            else:
                auc[f], acc[f], _, _ = self.model_eval(
                    parameter_hat[f, 0:nParam[-1], :],
                    data_test, range(len(self.channels)))
                count += 1
                # Set current status
                pb.set_stat(count)
                # Update Progress Bar again
                pb.update()

        # Make the Progress Bar final
        pb.end()
        if self.paradigm == "FRP":
            parameters = np.zeros((len(self.channels), 2 * nParam[-1]))
        else:
            parameters = np.zeros((len(self.channels), nParam[-1]))
        # Pick best parameters for each channel/brain source
        for ch in range(len(self.channels)):
            best_ind = [i for i in range(nFold) if
                        auc_ch[i, ch] == max(auc_ch[:, ch])]
            if self.paradigm == "FRP":
                parameters[ch, :] = parameter_hat[best_ind[0], 0:2 * nParam[-1],
                                    ch]
            else:
                parameters[ch, :] = parameter_hat[best_ind[0], 0:nParam[-1], ch]

        hyperParam = [self.order_AR] + self.tau + self.delays + list(
            self.compOrder)

        return auc, acc, parameters, hyperParam

    def data_for_corssValidation(self, data, nFold=10):
        """
            Generates K dictionary dataset for cross validation
            Input Args:
                data: a dictionary including timeseries, onsets of stimuli,
                        and targets for all sequences
                nFold: number of folds(int)
            Return:
                y_test: a list including timeseries signal for all test folds
                y_train: a list including timeseries signal for all training folds
                us_test: a list of onsets of stimuli across all test folds
                us_train: a list of onsets of stimuli across all training folds
                ue_test: a list of target locations across all test folds
                ue_train: a list of target locations across all training folds
        """
        # Initilization
        foldSampSize = np.floor(self.numSeq / nFold)
        eeg = data["timeseries"]
        trigOnsets = data["stimOnset"]
        targetOnsets = data["targetOnset"]
        # shuffling data set
        indx = np.random.permutation(self.numSeq)
        eeg_sh = eeg[:, :, indx]
        trigOnsets_sh = trigOnsets[:, indx]
        targetOnsets_sh = targetOnsets[:, indx]
        y_train = []
        y_test = []
        us_train = []
        us_test = []
        ue_train = []
        ue_test = []

        if self.paradigm == "FRP":
            ind_frp_neg = [i for i in range(self.numSeq) if
                           targetOnsets_sh[0, i] > 0]
            ind_frp_pos = [i for i in range(self.numSeq) if
                           targetOnsets_sh[0, i] == 0]
            y_train_pos = eeg_sh[:, :, ind_frp_pos]
            y_train_neg = eeg_sh[:, :, ind_frp_neg]
            us_train_pos = trigOnsets_sh[:, ind_frp_pos]
            us_train_neg = trigOnsets_sh[:, ind_frp_neg]
            ue_train_pos = targetOnsets_sh[:, ind_frp_pos]
            ue_train_neg = targetOnsets_sh[:, ind_frp_neg]
            y_train_pos_ = copy(y_train_pos)
            y_train_neg_ = copy(y_train_neg)
            us_train_pos_ = copy(us_train_pos)
            us_train_neg_ = copy(us_train_neg)
            ue_train_pos_ = copy(ue_train_pos)
            ue_train_neg_ = copy(ue_train_neg)
        else:
            y_train_ = copy(eeg_sh)
            us_train_ = copy(trigOnsets_sh)
            ue_train_ = copy(targetOnsets_sh)

        # training the model within K-fold cross validation
        for f in range(nFold):
            # This means you have only one fold and do want to cross validate
            if foldSampSize >= self.numSeq:
                if self.paradigm == "FRP":
                    y_test_pos = copy(y_train_pos)
                    y_test_neg = copy(y_train_neg)
                    us_test_pos = copy(us_train_pos)
                    us_test_neg = copy(us_train_neg)
                    ue_test_pos = copy(ue_train_pos)
                    ue_test_neg = copy(ue_train_neg)
                    y_train.append(
                        np.concatenate((y_train_pos, y_train_neg), 2))
                    us_train.append(
                        np.concatenate((us_train_pos, us_train_neg), 1))
                    ue_train.append(
                        np.concatenate((ue_train_pos, ue_train_neg), 1))
                    y_test.append(np.concatenate((y_test_pos, y_test_neg), 2))
                    us_test.append(
                        np.concatenate((us_test_pos, us_test_neg), 1))
                    ue_test.append(
                        np.concatenate((ue_test_pos, ue_test_neg), 1))
                else:
                    y_test.append(y_train_)
                    us_test.append(us_train_)
                    ue_test.append(ue_train_)
            else:
                if self.paradigm == "FRP":
                    testIndx = range(int(f * np.floor(foldSampSize / 2)),
                                     int((f + 1) * np.floor(foldSampSize / 2)))
                    testIndx_neg = testIndx
                    testIndx_pos = testIndx
                    indx_neg = [ti for ti in range(len(testIndx)) if
                                testIndx[ti] > len(ind_frp_neg) - 1]
                    indx_pos = [ti for ti in range(len(testIndx)) if
                                testIndx[ti] > len(ind_frp_pos) - 1]

                    if indx_neg:
                        testIndx_neg = testIndx[:indx_neg[0]]
                    if indx_pos:
                        testIndx_pos = testIndx[:indx_pos[0]]

                    y_test_pos = copy(y_train_pos_[:, :, testIndx_pos])
                    y_test_neg = copy(y_train_neg_[:, :, testIndx_neg])
                    us_test_pos = copy(us_train_pos_[:, testIndx_pos])
                    us_test_neg = copy(us_train_neg_[:, testIndx_neg])
                    ue_test_pos = copy(ue_train_pos_[:, testIndx_pos])
                    ue_test_neg = copy(ue_train_neg_[:, testIndx_neg])
                    y_test.append(np.concatenate((y_test_pos, y_test_neg), 2))
                    us_test.append(
                        np.concatenate((us_test_pos, us_test_neg), 1))
                    ue_test.append(
                        np.concatenate((ue_test_pos, ue_test_neg), 1))
                    tmp = copy(y_train_pos_)
                    y_train_pos = np.delete(tmp, testIndx_pos, 2)
                    tmp = copy(y_train_neg_)
                    y_train_neg = np.delete(tmp, testIndx_neg, 2)
                    tmp = copy(us_train_pos_)
                    us_train_pos = np.delete(tmp, testIndx_pos, 1)
                    tmp = copy(us_train_neg_)
                    us_train_neg = np.delete(tmp, testIndx_neg, 1)
                    tmp = copy(ue_train_pos_)
                    ue_train_pos = np.delete(tmp, testIndx_pos, 1)
                    tmp = copy(ue_train_neg_)
                    ue_train_neg = np.delete(tmp, testIndx_neg, 1)
                    y_train.append(
                        np.concatenate((y_train_pos, y_train_neg), 2))
                    us_train.append(
                        np.concatenate((us_train_pos, us_train_neg), 1))
                    ue_train.append(
                        np.concatenate((ue_train_pos, ue_train_neg), 1))
                else:
                    testIndx = range(int(f * np.floor(foldSampSize)),
                                     int((f + 1) * np.floor(foldSampSize)))
                    y_test.append(y_train_[:, :, testIndx])
                    us_test.append(us_train_[:, testIndx])
                    ue_test.append(ue_train_[:, testIndx])
                    tmp = copy(y_train_)
                    y_train.append(np.delete(tmp, testIndx, 2))
                    tmp = copy(us_train_)
                    us_train.append(np.delete(tmp, testIndx, 1))
                    tmp = copy(ue_train_)
                    ue_train.append(np.delete(tmp, testIndx, 1))

        return y_test, y_train, us_test, us_train, ue_test, ue_train

    def model_eval(self, parameter, data, channel):
        """
            Evaluate the estimated parameters for ARX model using loglikelihood
            measure.
            Input Args:
                parameters: model parameters for each channel/brain sources
                data: a dictionary including timeseries, onsets of stimuli,
                    and targets for all sequences
                channel: set of channels/brain sources (int)
            Return:
                auc: AUC of the classifier
                acc: accuracy of the classifier
                scores: loglikelihood scores of of all trials
                trialTargetness: represents targetness of each trial across
                                 all sequences
        """
        numSeq = data["numSeq"]
        y = data["timeseries"]
        us = data["stimOnset"]
        ue = data["targetOnset"]
        # Initialization
        label = np.zeros((numSeq, 1))
        coeff = data["coeff"]
        score = np.zeros((len(channel), numSeq * (self.numTrial + 1)))
        if self.paradigm == "FRP":
            loglikelihood = np.zeros((numSeq, 2, len(channel)))
            trialTargetness = np.zeros((numSeq, 2))
        else:
            loglikelihood = np.zeros((numSeq, self.numTrial + 1, len(channel)))
            trialTargetness = np.zeros((numSeq, self.numTrial + 1))

        # Computing loglikehood scores for each possible target location
        for seq in range(numSeq):
            if self.paradigm == "FRP":
                sc = np.zeros((len(channel), numSeq))
                # negative sequences
                if ue[0, seq] > 0:
                    label[seq, 0] = 1
                    trialTargetness[seq, 1] = 1
                    trialTargetness[seq, 0] = 0
                # positive sequences
                else:
                    label[seq, 0] = 0
                    trialTargetness[seq, 1] = 0
                    trialTargetness[seq, 0] = 1

                len_param = parameter.shape[0] / 2
                # compute loglikelihood score for positive sequences
                loglikelihood[seq, 0, :] = self.loglikelihoodARX(
                    y[channel, :, seq],
                    us[:, seq], [], parameter[:len_param, :])
                # compute loglikelihood score for negative sequences
                loglikelihood[seq, 1, :] = self.loglikelihoodARX(
                    y[channel, :, seq],
                    us[:, seq], [], parameter[len_param:, :])
            else:
                sc = np.zeros((self.numTrial + 1, numSeq))
                targetLoc = ue[0, seq]
                possibleTarget = np.concatenate(([0], us[:, seq]))
                for trial in range(self.numTrial + 1):
                    if np.abs(possibleTarget[trial] - targetLoc) < 3:
                        label[seq] = trial + 1
                        trialTargetness[seq, trial] = 1
                    else:
                        trialTargetness[seq, trial] = 0
                    loglikelihood[seq, trial, :] = self.loglikelihoodARX(
                        y[channel, :, seq], us[:, seq],
                        possibleTarget[trial], parameter)
        # Computing auc and acc of the classifier
        if self.paradigm == "FRP":
            for ch in range(len(channel)):
                sc[ch, :] = np.matmul(np.array([[-1., 1.]]),
                                      (-1) * loglikelihood[:, :, ch].T)

            if len(channel) > 1:
                # combining multi-channels/brain sources
                scores = np.matmul(coeff, sc)
            else:
                scores = copy(sc[0])
                scores = np.expand_dims(scores, 0)[0, :]

            auc, acc, _, _ = self.calculateROC(scores, label[:, 0])
            trialTargetness = label[:, 0]

        else:
            for ch in range(len(channel)):
                ss = (-1) * loglikelihood[:, :, ch]
                for seq in range(numSeq):
                    sc[:, seq] = ss[seq, :] - ss[
                        seq, trialTargetness[seq, :] > 0]
                    sc[sc[:, seq] == 0, seq] = (-1) * np.mean(
                        sc[sc[:, seq] != 0, seq])

                score[ch, :] = np.reshape(sc, (self.numTrial + 1) * numSeq, 1)

            if len(channel) > 1:
                scores = np.matmul(score.T, coeff)
            else:
                scores = copy(score[0])

            t_target = np.reshape(trialTargetness, (self.numTrial + 1) * numSeq,
                                  1)
            auc, _, _, _ = self.calculateROC(scores, t_target)
            ss = np.reshape(scores, ((self.numTrial + 1), numSeq))
            tt = np.reshape(t_target, ((self.numTrial + 1), numSeq))
            acc = self.calculateACC(ss, tt)
            trialTargetness = []
            trialTargetness = t_target

        return auc, acc, scores, trialTargetness

    def cyclic_decent_method(self, data, channel):
        """
            The cyclic descent algorithm is used for joint estimation f the
            AR(p) and design matrix coefficients.
            For more information check out the following papers:
            - W. Q. Malik, et al., "Denoising two-photon calcium imaging data",
              PloS one, 2011.
            - Y. M. Marghi, et al., "A Parametric EEG Signal Model for BCIs with
              Rapid-Trial Sequences" EMBC, 2018.

            Input Args:
                data: a dictionary including timeseries, onsets of stimuli,
                      and targets for all sequences
                channel: set of channels/brain sources (int)
            Return:
                parameter: estimated parameters fpr the ARX model including AR
                           parameters, gammafunction

                loglikelihood: loglikelihood scores
                sigma_hat: estimation error
                y_hat: synthetic eeg signal
                signal_hat: estimated brain activities due to visual stimuli
                            including VEP+ERP
                s_hat: estimated AR process in the signal model
        """
        # Initialization
        eeg = data["timeseries"][channel, :, :]
        trigOnsets = data["stimOnset"]
        numSeq = data["numSeq"]
        N = self.num_sample - self.order_AR
        Ix = np.identity(N)
        err, b = self.arburg(eeg[:, 0])
        var_hat = .1 * err
        var_hat_old = 1.5 * err
        Q_inv = copy(Ix)
        Q_inv_nonFRP = copy(Ix)
        Q_inv_FRP = copy(Ix)
        iter = 0
        parameter = []
        s_hat = np.zeros((self.num_sample, numSeq))
        X = []  # design matrix
        # Parameter estimation using cyclic decent method
        while abs(var_hat - var_hat_old) > self.threshold * abs(var_hat_old):
            iter += 1
            var_hat_old = var_hat
            sig_hat = np.zeros((self.num_sample, numSeq))
            ar_hat = np.zeros((self.num_sample, numSeq))
            X = []
            y = []
            y_neg = []
            y_pos = []
            arProcess_hat = []
            Q_inv_s = []
            n_neg = 0
            n_pos = 0
            z = np.zeros(
                (np.sum(self.compOrder), self.num_sample - self.order_AR))
            ar_hat = np.zeros((self.num_sample, numSeq))
            if self.paradigm == "FRP":
                beta_s = np.zeros((np.sum(self.compOrder), 2))
            else:
                beta_s = np.zeros((np.sum(self.compOrder), 1))

            for s in range(numSeq):
                vep = np.zeros((self.compOrder[0], N))
                for trial in range(self.numTrial):
                    input = trigOnsets[trial, s] + self.delays[0]
                    vep += self.gammafunction(input, self.compOrder[0],
                                              self.tau[0])

                z[0:self.compOrder[0], :] = vep
                if self.paradigm == "FRP":
                    input_p = trigOnsets[0, s] + self.delays[1]
                    input_q = trigOnsets[0, s] + self.delays[2]
                    z[self.compOrder[0]:np.sum(self.compOrder[:2]), :] = \
                        self.gammafunction(input_p, self.compOrder[1],
                                           self.tau[1])
                    z[np.sum(self.compOrder[:2]):np.sum(self.compOrder), :] = \
                        self.gammafunction(input_q, self.compOrder[2],
                                           self.tau[2])
                else:
                    if data["targetOnset"][0][s] > 0:
                        input_p = data["targetOnset"][0][s] + self.delays[1]
                        input_q = data["targetOnset"][0][s] + self.delays[2]
                        tmp1 = self.gammafunction(input_p, self.compOrder[1],
                                                  self.tau[1])
                        tmp2 = self.gammafunction(input_q, self.compOrder[2],
                                                  self.tau[2])
                    else:
                        tmp1 = np.zeros((self.compOrder[1], N))
                        tmp2 = np.zeros((self.compOrder[2], N))

                    z[self.compOrder[0]:np.sum(self.compOrder[:2]), :] = tmp1
                    z[np.sum(self.compOrder[:2]):np.sum(self.compOrder),
                    :] = tmp2

                # updates the design matrix according to the target/non-target stimuli onsets
                X.append(z.T)
                y_s = eeg[self.order_AR:, s]

                # check the experiment paradigm
                if self.paradigm == "FRP":
                    # separate positive and negative FRP sequences
                    if data["targetOnset"][0][s] > 0:
                        # negative sequences
                        y_neg.append(y_s)
                        temp = np.matmul(z, Q_inv)
                        beta_s[:, 1] = np.matmul(
                            np.linalg.inv(np.matmul(temp, z.T)),
                            np.matmul(temp, y_s))
                        # estimated VEP & ERP terms
                        sig_hat[self.order_AR:, s] = np.matmul(z.T, beta_s[:, 1])
                    else:
                        # positive sequences
                        y_pos.append(y_s)
                        temp = np.matmul(z, Q_inv)
                        beta_s[:, 0] = np.matmul(
                            np.linalg.inv(np.matmul(temp, z.T)),
                            np.matmul(temp, y_s))
                        # estimated VEP & ERP terms
                        sig_hat[self.order_AR:, s] = np.matmul(z.T, beta_s[:, 0])

                if self.paradigm == "ERP":
                    temp = np.matmul(z, Q_inv)
                    beta_s = np.matmul(np.linalg.inv(np.matmul(temp, z.T)),
                                       np.matmul(temp, y_s))
                    # estimated VEP & ERP terms
                    sig_hat[self.order_AR:, s] = np.matmul(z.T, beta_s)
                    y.append(y_s)

                # AR series - is this ar_hat + res_hat
                arProcess_hat.append(y_s - sig_hat[self.order_AR:, s])
                w_inv, d_vec, q = self.invCov(arProcess_hat[-1], Ix, Ix)
                Q_inv_s.append(w_inv)
                Q_inv = copy(w_inv)

            matX = np.zeros((sum(self.compOrder), sum(self.compOrder)))
            matY = np.zeros((1, sum(self.compOrder)))
            matY_neg = np.zeros((1, sum(self.compOrder)))
            matY_pos = np.zeros((1, sum(self.compOrder)))
            n_neg = 0
            n_pos = 0
            err = []
            for s in range(numSeq):
                err.append(np.matmul(arProcess_hat[s].T,
                                     np.matmul(Q_inv, arProcess_hat[s])))
                matX += np.matmul(X[s].T, np.matmul(Q_inv_s[s], X[s]))
                # check the experiment paradigm
                if self.paradigm == "FRP":
                    # separate positive and negative FRP sequences
                    if data["targetOnset"][0][s] > 0:
                        # negative sequences
                        matY_neg += np.matmul(X[s].T,
                                              np.matmul(Q_inv_s[s],
                                                        y_neg[n_neg]))
                        n_neg += 1
                    else:
                        # positive sequences
                        matY_pos += np.matmul(X[s].T,
                                              np.matmul(Q_inv_s[s],
                                                        y_pos[n_pos]))
                        n_pos += 1
                if self.paradigm == "ERP":
                    matY += np.matmul(X[s].T, np.matmul(Q_inv_s[s], y[s]))

            # Design matrix parameters
            # check the experiment paradigm
            if self.paradigm == "FRP":
                # negative sequences
                beta_hat_neg = np.matmul(np.linalg.inv(matX), matY_neg.T)
                # positive sequences
                beta_hat_pos = np.matmul(np.linalg.inv(matX), matY_pos.T)
            if self.paradigm == "ERP":
                beta_hat = np.matmul(np.linalg.inv(matX), matY.T)

            # Update Q_inv with a detrended V so arburg works well
            arProcess_vect = np.reshape(arProcess_hat, (N * numSeq, 1))
            Q_inv, D, q = self.invCov(arProcess_vect[:, 0], Ix, Ix)
            # Estimate alpha and sigma2e with burg
            sigma_hat, ar_coeff = self.arburg(arProcess_vect[:, 0])
            error = np.mean(err)
            # Mean square error - diff. from mse of lscov
            var_hat = sigma_hat

        alpha_hat = np.expand_dims((-1) * ar_coeff[0, 1:], 1)
        sigma_hat = np.expand_dims(np.expand_dims(sigma_hat, 1), 1)
        if self.paradigm == "FRP":
            parameter_pos = np.concatenate((alpha_hat, beta_hat_pos, sigma_hat))
            parameter_neg = np.concatenate((alpha_hat, beta_hat_neg, sigma_hat))
            parameter = np.concatenate((parameter_pos, parameter_neg))

        if self.paradigm == "ERP":
            parameter = np.concatenate((alpha_hat, beta_hat, sigma_hat))

        # Compute the loglikelihood
        logdG = np.sum(np.log(D)) + N * np.log(sigma_hat)
        loglikelihood = numSeq * logdG + error / sigma_hat
        # Estimated EEG sequences based on the estimated parameters
        for s in range(numSeq):
            s_tilde = eeg[:self.order_AR, s]
            noise = np.sqrt(sigma_hat) * np.random.randn(self.num_sample)
            s_hat[:self.order_AR, s] = noise[0, :self.order_AR]
            for n in range(self.order_AR, self.num_sample):
                s_hat[n, s] = np.matmul(np.flip(s_tilde, 0).T, alpha_hat) + \
                              noise[0, n]
                s_tilde = np.concatenate(
                    (s_tilde[1:], np.expand_dims(s_hat[n, s], 0)))

        signal_hat = np.reshape(sig_hat, (self.num_sample, numSeq))
        y_hat = signal_hat + s_hat

        return parameter, loglikelihood, sigma_hat, y_hat, signal_hat, s_hat

    def syntheticEEGseq(self, parameter, trigOnsets, targetOnset):
        """
            Generates synthetic EEG signals according to the parameters
            learning in the 'modelfitting' mode
            Input Args:
                parameter: a vector of the ARX model parameters
                trigOnsets: trigger information for all trials, 2D matrix,
                           numberTrial x numberSequence
                targetOnset: trigger information for target events, vector,
                             1 x numberSequence
            Return:
                syn_data: synthetic data, a dictionary
        """
        N = self.num_sample - self.order_AR
        Ix = np.identity(N)
        numSeq = targetOnset.shape[1]
        s_hat = np.zeros((self.num_sample, numSeq))
        eeg_hat = np.zeros((len(self.channels), self.num_sample, numSeq))
        nParam = [self.order_AR, self.order_AR + sum(self.compOrder),
                  self.order_AR + sum(self.compOrder) + 1]
        # generate multi-channel EEG signals for M sequences
        # Create Object
        prog = pyprog.ProgressBar(" ", "", total=numSeq)
        # Update Progress Bar
        prog.update()
        for s in range(numSeq):
            # generate signal for each channel
            for ch in range(len(self.channels)):
                z = np.zeros(
                    (np.sum(self.compOrder), self.num_sample - self.order_AR))
                # VEP components
                vep = np.zeros((self.compOrder[0], N))
                for trial in range(trigOnsets.shape[0]):
                    input = trigOnsets[trial, s] + self.delays[0]
                    vep += self.gammafunction(input, self.compOrder[0],
                                              self.tau[0])
                z[0:self.compOrder[0], :] = vep
                if self.paradigm == "FRP":
                    input_p = trigOnsets[0, s] + self.delays[1]
                    input_q = trigOnsets[0, s] + self.delays[2]
                    z[self.compOrder[0]:np.sum(self.compOrder[:2]), :] = \
                        self.gammafunction(input_p, self.compOrder[1],
                                           self.tau[1])
                    z[np.sum(self.compOrder[:2]):np.sum(self.compOrder), :] = \
                        self.gammafunction(input_q, self.compOrder[2],
                                           self.tau[2])
                    # separate parameters for -FRP and +FRP
                    len_param = parameter.shape[1] / 2
                    if targetOnset[0, s] > 0:
                        param = parameter[ch, len_param:]
                    else:
                        param = parameter[ch, :len_param]
                    alpha = param[:nParam[0]]
                    beta = param[nParam[0]:nParam[1]]
                    sigma_hat = param[-1]
                else:
                    if targetOnset[0, s] > 0:
                        input_p = targetOnset[0, s] + self.delays[1]
                        input_q = targetOnset[0, s] + self.delays[2]
                        tmp1 = self.gammafunction(input_p, self.compOrder[1],
                                                  self.tau[1])
                        tmp2 = self.gammafunction(input_q, self.compOrder[2],
                                                  self.tau[2])
                    else:
                        tmp1 = np.zeros((self.compOrder[1], N))
                        tmp2 = np.zeros((self.compOrder[2], N))

                    z[self.compOrder[0]:np.sum(self.compOrder[:2]), :] = tmp1
                    z[np.sum(self.compOrder[:2]):np.sum(self.compOrder),
                    :] = tmp2
                    alpha = parameter[ch, :nParam[0]]
                    beta = parameter[ch, nParam[0]:nParam[1]]
                    sigma_hat = parameter[ch, -1]
                # estimated VEP & ERP/FRP terms
                sig_hat = np.concatenate(
                    (np.zeros((self.order_AR)), np.matmul(z.T, beta)))
                # Estimated EEG sequences based on the estimated parameters
                s_tilde = np.zeros((self.order_AR, 1))
                noise = np.sqrt(sigma_hat) * np.random.randn(self.num_sample)
                s_hat[:self.order_AR, s] = noise[:self.order_AR]
                for n in range(self.order_AR, self.num_sample):
                    s_hat[n, s] = np.matmul(np.flip(s_tilde, 0).T, alpha) + \
                                  noise[n]
                    s_tilde[:, 0] = np.concatenate(
                        [s_tilde[1:, 0], np.expand_dims(s_hat[n, s], 0)])

                signal_hat = np.reshape(sig_hat, (self.num_sample, 1))
                eeg_hat[ch, :, s] = signal_hat[:, 0] + s_hat[:, s]
            # Set current status
            prog.set_stat((s + 1.))
            # Update Progress Bar again
            prog.update()

        # Make the Progress Bar final
        prog.end()
        syn_data = dict()
        syn_data.update({"timeseries": eeg_hat,
                         "stimOnset": trigOnsets,
                         "targetOnset": targetOnset})
        return syn_data

    def gammafunction(self, x, compOrder, tau):
        """
            Generates the brain impulse responses to the visual and this
            response can be modeled by the gamma function  as follow:

                f(n,tau,L) = sum_l(b_l*(n/l)^l*exp(-n/tau)) u[n]

            for more information, please check equation (4) in "A Parametric
            EEG Signal Model for BCIs with Rapid-Trial Sequences" paper.
            Input Args:
                x: a time series comes from AR process, Nx1
                compOrder: order of the polynomial exponential function
                tau: indicates skewness of the exponential function
            Return:
                y_gamma: an array including the value of gamma function
        """
        y = []
        try:
            dn = np.min(np.diff(x))
        except:
            dn = 0

        try:
            N = x.shape[0]
            Onset0 = x[0]
        except:
            N = 1
            Onset0 = x

        for s in range(N):
            M = self.num_sample - Onset0 - s * dn + 1
            m = np.array(range(int(M)))
            z = []  # np.zeros((comp_order, M))

            for r in range(compOrder):
                z.append(np.power((m / float(r + 1)), (r + 1)) * np.exp(
                    -1. * m / tau))

            if dn == 0:
                y += z
            else:
                temp = np.zeros((compOrder, s * dn))
                z = z.reshape()
                y += np.concatenate((temp, z), 0)

        temp = np.zeros((compOrder, int(Onset0 - self.order_AR - 1)))
        y_gamma = np.concatenate((temp, y), 1)

        return y_gamma

    def invCov(self, x, L_inv, w_inv):
        """
            Calculate inverse of AR covariance matrix via LDR V is NXps ensemble
            of residuals, Linv & Winv are initialized to I_N
            Input Args:
                x: a time series comes from AR process, Nx1
                L_inv: order of the polynomial exponential function
                w_inv: inverse covariance of AR(p)
            Return:
                w_inv: vectorized inverse covariance of AR(p) - Cholesky Form
                d_vec: elements of diagonal matrix as per Levinson-Durbin recursion
                q: error(p) as a output of arburg(x,P)
        """
        # Find AR coefficients
        err, b = self.arburg_(x)
        L_invt = copy(L_inv)
        N = self.num_sample - self.order_AR
        # Transpose of L_invt matrix as per Levinson-Durbin recursion
        for ar in range(self.order_AR):
            coef = b[ar]
            L_invt[:ar + 1, ar + 1] = coef

        for ar in range(N - self.order_AR - 1):
            L_invt[ar + 1:ar + self.order_AR + 1, ar + self.order_AR] = coef

        # Apply Levinson-Durbin recursion to get the diagonal elements
        x_xcov = np.cov(x, bias=False, ddof=0)
        d_vec = np.concatenate((np.array([[x_xcov]]), err), 0)
        d_inv = float(1) / d_vec
        q = err[-1]
        q_ = float(1) / q
        # Compute inverse covariance of AR(p) using Cholesky Form
        A = L_invt[:self.order_AR + 1, :self.order_AR + 1]
        B = L_invt[:self.order_AR + 1, self.order_AR + 1:self.num_sample]
        C = L_invt[self.order_AR + 1:self.num_sample,
            self.order_AR + 1:self.num_sample]
        D = np.diag(d_inv[:, 0])
        AD = np.matmul(A, D)
        w_inv = copy(w_inv)
        w_inv[:self.order_AR + 1, :self.order_AR + 1] = \
            np.matmul(AD, A.T) + q_ * np.matmul(B, B.T)
        w_inv[:self.order_AR + 1,
        self.order_AR + 1:self.num_sample] = q_ * np.matmul(B, C.T)
        w_inv[self.order_AR + 1:self.num_sample,
        :self.order_AR + 1] = q_ * np.matmul(C, B.T)
        w_inv[self.order_AR + 1:self.num_sample,
        self.order_AR + 1:self.num_sample] = q_ * np.matmul(C, C.T)

        return w_inv, d_vec, q

    def loglikelihoodARX(self, y, us, ue, parameter):
        """
            Computes the loglikelihood scores
            Input Args:
                y: time series of multi channel/brain sources measurement
                us: trigger information of SSVEPs within sequences
                ue: trigger information of the target event in the sequence
                parameters: model parameters for each channel/brain sources
            Return:
                log_score: loglikelihood scores(s)
        """
        # Initialization
        N = self.num_sample - self.order_AR
        Ix = np.identity(N)
        z = np.zeros((np.sum(self.compOrder), self.num_sample - self.order_AR))
        log_score = np.zeros((y.shape[0]))
        nParam = [self.order_AR, self.order_AR + sum(self.compOrder),
                  self.order_AR + sum(self.compOrder) + 1]
        # VEP components
        vep = np.zeros((self.compOrder[0], N))
        for trial in range(self.numTrial):
            input = us[trial] + self.delays[0]
            vep += self.gammafunction(input, self.compOrder[0], self.tau[0])

        z[0:self.compOrder[0], :] = vep

        if self.paradigm == "FRP":
            input_p = us[0] + self.delays[1]
            input_q = us[0] + self.delays[2]
            z[self.compOrder[0]:np.sum(self.compOrder[:2]), :] = \
                self.gammafunction(input_p, self.compOrder[1], self.tau[1])
            z[np.sum(self.compOrder[:2]):np.sum(self.compOrder), :] = \
                self.gammafunction(input_q, self.compOrder[2], self.tau[2])
            # compute scores for each channel
            for ch in range(y.shape[0]):
                beta = parameter[nParam[0]:nParam[1], ch]
                sigma_hat = parameter[-1, ch]
                # estimated VEP & FRP terms
                sig_hat = np.matmul(z.T, beta)
                # AR series - is this ar_hat + res_hat
                arProcess_hat = y[ch, self.order_AR:] - sig_hat
                Q_inv, D, _ = self.invCov(arProcess_hat, Ix, Ix)
                error = np.matmul(arProcess_hat.T, np.matmul(Q_inv,
                                                             arProcess_hat))
                logdG = np.sum(np.log(D)) + N * np.log(sigma_hat + eps)
                log_score[ch] = logdG + error / (sigma_hat + eps)

        else:
            if ue > 0:
                input_p = ue + self.delays[1]
                input_q = ue + self.delays[2]
                tmp1 = self.gammafunction(input_p, self.compOrder[1],
                                          self.tau[1])
                tmp2 = self.gammafunction(input_q, self.compOrder[2],
                                          self.tau[2])
            else:
                tmp1 = np.zeros((self.compOrder[1], N))
                tmp2 = np.zeros((self.compOrder[2], N))

            z[self.compOrder[0]:np.sum(self.compOrder[:2]), :] = tmp1
            z[np.sum(self.compOrder[:2]):np.sum(self.compOrder), :] = tmp2
            # compute scores for each channel
            for ch in range(y.shape[0]):
                beta = parameter[nParam[0]:-1, ch]
                sigma_hat = parameter[-1, ch]
                # estimated VEP & ERP terms
                sig_hat = np.matmul(z.T, beta)
                # AR series - is this ar_hat + res_hat
                arProcess_hat = y[ch, self.order_AR:] - sig_hat
                Q_inv, D, _ = self.invCov(arProcess_hat, Ix, Ix)
                error = np.matmul(arProcess_hat.T, np.matmul(Q_inv,
                                                             arProcess_hat))
                logdG = np.sum(np.log(D)) + N * np.log(sigma_hat + eps)
                log_score[ch] = logdG + error / (sigma_hat + eps)

        return log_score

    def multiChanenelCoeff(self, scores, label, nFold=10):
        """
            Takes scores from ch number of channel or brain sources and N number
            of trials and compute the coefficients of Lasso Logistic Regression.
            Input Args:
                scores: classifier scores
                labels: true labels
                nFold: number of folds for cross validation
            Return:
                auc_mean: mean of AUC values across n folds after learning
                          hyperparamter
                auc_std: std of AUC values across n folds after learning
                        hyperparamter
                coeff: coefficient of determination R^2 of the prediction
                alp: hyperparameter of Ridge regression
        """
        # Initialization
        auc_mean = []
        auc_std = []
        # Shuffling scores
        n = scores.shape[0]
        indx = np.random.permutation(n)
        score_sh = scores[indx, :]
        label_sh = label[indx]
        k = [i * .01 for i in range(100)]
        k = k + [2., 5., 10.]
        # Fold creation
        stp = int(np.floor(n / nFold))

        for j in k:
            auc = []
            for i in range(nFold):
                dummy_score = score_sh
                dummy_label = label_sh
                if i == nFold:
                    indx = [q for q in range(i * stp, n)]
                    score_test = dummy_score[indx, :]
                    label_test = dummy_label[indx]
                    np.delete(dummy_score, indx, 0)
                    np.delete(dummy_label, indx, 0)
                else:
                    indx = [q for q in range(i * stp, (i + 1) * stp)]
                    score_test = dummy_score[indx, :]
                    label_test = dummy_label[indx]
                    np.delete(dummy_score, indx, 0)
                    np.delete(dummy_label, indx, 0)

                score_train = dummy_score
                label_train = dummy_label
                # Apply ridge regression to estimate the coefficients
                clf = []
                clf = linear_model.Ridge(alpha=j)
                clf.fit(score_train, label_train)
                coeff = clf.coef_
                score_pred = np.matmul(score_test, coeff)
                # Compute AUC
                tmp, _, _, _ = self.calculateROC(score_pred, label_test)
                auc.append(tmp)
            # Compute mean and std of mean over n folds
            auc_ = [a for a in auc if ~np.isnan(a)]
            auc_mean.append(np.mean(auc_))
            auc_std.append(np.std(auc_))
        # Find the best regularization parameter (hyperparamter)
        indx = [q for q in range(len(k)) if auc_mean[q] == max(auc_mean)]
        alp = k[indx[0]]
        # Apply again ridge regression with the learned hyperparamter
        coeff = []
        clf = []
        clf = linear_model.Ridge(alpha=alp)
        clf.fit(score_sh, label_sh)
        coeff = clf.coef_

        return max(auc_mean), auc_std[indx[0]], coeff, alp

    def arburg(self, x):
        """
            AR parameter estimation via Burg method.
            A = ARBURG(X,ORDER) returns the coefficients of the AR parametric
            signal model estimate of X using Burg's method. The model has order
            ORDER, and the output array A has ORDER+1 columns.
            The coefficients along the Nth row of A model the Nth column of X.
            If X is a vector then A is a row vector.

            [A,E] = ARBURG(...) returns the final prediction error E (the variance
            estimate of the white noise input to the AR model).

            [A,E,K] = ARBURG(...) returns the reflection coefficients (parcor
            coefficients) in each column of K.
            Input Args:
                x: a time series comes from AR process, Nx1
            Return:
                err: final prediction error (estimated white noise variance)
                k: has size 1xm and m can range from 1 to AR order, error is mXps
        """

        # Initialization
        N = float(x.shape[0])
        a = [[1]]
        err = np.sum(x * x) / N
        k = np.zeros((self.order_AR, 1))
        efp = x[1:]
        ebp = x[:-1]
        # Burg algorithm recursions
        for ar in range(self.order_AR):
            # Calculate the next order reflection (parcor) coefficient
            num = float(-2) * np.sum(ebp * efp)
            den = np.sum(efp * efp) + np.sum(ebp * ebp)
            kk = num / den
            k[ar] = kk
            # Update the forward and backward prediction errors
            ef = efp[1:] + kk * ebp[1:]
            ebp = ebp[:-1] + kk * efp[:-1]
            efp = ef
            # Update the AR coefficients
            a_flip = np.flip(a, 1)
            a_conj = a_flip.conjugate(a_flip)
            a = np.concatenate((a, [[0]]), 1) + kk * np.concatenate(
                ([[0]], a_conj), 1)
            # Update the prediction error
            err = (1 - kk * kk) * err

        return err, a

    def arburg_(self, x):
        """
            This function Vectorized AR parameter estimation via Burg method
            Input Args:
                x: a time series comes from AR process, Nx1
            Return:
                err: final prediction error (estimated white noise variance)
                b{m}: has size 1xm and m can range from 1 to AR order, error is mXps
        """

        # Initialization
        N = float(x.shape[0])
        ef = x
        eb = x
        b = []
        a = [[1]]
        err = np.zeros((self.order_AR + 1, 1))
        err[0] = np.sum(x * x) / N
        k = np.zeros((self.order_AR, 1))

        # Burg algorithm recursions
        for ar in range(self.order_AR):
            # Calculate the next order reflection (parcor) coefficient
            efp = ef[1:]
            ebp = eb[:-1]
            num = float(-2) * np.sum(ebp * efp)
            den = np.sum(efp * efp) + np.sum(ebp * ebp)
            kk = num / den
            k[ar] = kk
            # Update the forward and backward prediction errors
            ef = efp + kk * ebp
            eb = ebp + kk * efp
            # Update the AR coefficients
            a_flip = np.flip(a, 1)
            a_conj = a_flip.conjugate(a_flip)
            a = np.concatenate((a, [[0]]), 1) + kk * np.concatenate(
                ([[0]], a_conj), 1)
            coef = float(-1) * a[0][1:]
            coef = float(-1) * np.flip([coef], 1)
            b.append(coef)
            # Update the prediction error
            err[ar + 1] = (1 - kk * kk) * err[ar]

        err = err[1:]

        return err, b

    def calculateROC(self, scores, labels):
        """
        C   omputes the area under the ROC curve (AUC) of a binary classifier.
            Input Args:
                scores: classifier scores
                labels: true labels
            Return:
                auc: the area under the ROC curve
                acc: accuracy, number of correct assessments divided by number
                     of all samples
                sensitivity: number of true positive assessments divided by
                             number of all positive samples
                specificity: number of true negative assessments divided by
                             number of all negative samples
        """
        fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
        fnr, tnr, _ = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)
        acc = (np.sum(tpr) + np.sum(tnr)) / len(labels)
        sensitivity = np.sum(tpr) / len(labels[labels > 0])
        specificity = np.sum(tnr) / len(labels[labels == 0])

        return auc, acc, sensitivity, specificity

    def calculateACC(self, scores, labels):
        """
            Computes the accuracy of the classification.
            Input Args:
                scores: classifier scores
                labels: true labels
            Return:
                acc: accuracy
        """
        # Initialization
        err_sum = 0.0
        numSamp = scores.shape[1]
        # Predicting labels according to the scores
        for s in range(numSamp):
            sc = scores[:, s]
            indx = [i for i in range(self.numTrial + 1) if sc[i] == min(sc)]
            sc = np.zeros(self.numTrial + 1)
            sc[indx[0]] = 1
            err_sum += np.sum(np.abs(sc - labels[:, s]))

        acc = 1 - err_sum / (numSamp * (self.numTrial + 1))

        return acc
