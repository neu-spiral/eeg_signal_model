import numpy as np
import warnings
from copy import copy
from sklearn import metrics
import pyprog


class ARXmodelfit(object):
    """ Given the EEG data and visual stimuli information across sequences, this class fits an ARX model to EEG signals

        Attr:
            ....

        Functions:
            sequence_model():
                t....
            model_eval():
                ...
            cyclic_decent_method():

            gammafunction():

            invCov():

            loglikelihood():

            auc():
        """

    def __init__(self, fs, paradigm, numTrial, numSeq, numSamp, hyperparameter,
                 channels, threshold=1e-6, orderSelection=False):
        self.fs = fs
        self.paradigm = paradigm
        self.channels = channels
        self.numSamp = numSamp
        self.numSeq = numSeq
        self.numTrial = numTrial

        if orderSelection:
            self.ARorder = hyperparameter[:,0]
            self.tau = hyperparameter[:,1:4]
            self.delays = hyperparameter[:,4:7]
            self.compOrder = hyperparameter[0,7:]
        else:
            self.ARorder = hyperparameter[0]
            self.tau = hyperparameter[1:4]
            self.delays = hyperparameter[4:7]
            self.compOrder = hyperparameter[7:]

        self.threshold = threshold
        self.orderSelection = orderSelection


    def ARXmodelfit(self, data, nFold = 10):
        """
        fits an ARX model to the multi-channel/ brain sources timeseries signal
        using the cyclic decent method
            Input Args:
                data: a dictionary including timeseries, onsets of stimuli,
                        and targets for all sequences
                nFold: number of folds(int)
                orderSelection: True if you want to set hyperparmeters by BIC
                                and grid search (boolean)
            Return:
                auc: AUC of the classifier at each fold (nFold x 1)
                acc: accuracy of the classifier at each fold (nFold x 1)
                parameters: best parameters for each channel/brain sources
        """
        eeg = data["timeseries"]
        trigOnsets = data["stimOnset"]
        targetOnsets = data["targetOnset"]
        # Initialization
        auc_ch = np.zeros((nFold, len(self.channels)))
        auc = np.zeros((nFold, 1))
        acc = np.zeros((nFold, 1))
        parameter_hat = np.zeros((nFold, len(self.channels)))
        parameter_ = []
        trialTargetness = []
        score = []

        # Time resolution for delays in grid search for model order selection
        if self.orderSelection:
            dD = np.round(.005*float(self.fs))
            delay0 = [int(self.delays[0][0]+i*dD) for i in
                      range(int((self.delays[1][0] - self.delays[0][0])*dD+1))]
            delay1 = [int(self.delays[0][1]+i*dD) for i in
                      range(int((self.delays[1][1] - self.delays[0][1])*dD+1))]
            delay2 = [int(self.delays[0][2]+i*dD) for i in
                      range(int((self.delays[1][2] - self.delays[0][2])*dD+1))]
            AR_range = range(self.ARorder[0],self.ARorder[1]+1)
            tau0_range = range(self.tau[0][0],self.tau[1][0]+1)
            tau1_range = range(self.tau[0][1],self.tau[1][1]+1)
            tau2_range = range(self.tau[0][2],self.tau[1][2]+1)

        # training the model within K-fold cross validation
        y_test, y_train, us_test, us_train, ue_test, ue_train = \
            self.data_for_corssValidation(data, nFold)

        # Create Object
        prog = pyprog.ProgressBar(" ", "", 34)
        # Update Progress Bar
        prog.update()
        for f in range(nFold):
            data_train = dict()
            data_test = dict()
            data_train.update({"timeseries":y_train[f], "stimOnset":us_train[f],
                          "targetOnset":ue_train[f], "numSeq":ue_train[f].shape[1]})
            data_test.update({"timeseries":y_test[f], "stimOnset":us_test[f],
                         "targetOnset":ue_test[f], "numSeq":ue_test[f].shape[0]})
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
                        for tau0 in tau0_range:
                            for tau1 in tau1_range:
                                for tau2 in tau2_range:
                                    for d0 in delay0:
                                        for d1 in delay1:
                                            for d2 in delay2:
                                                nParam = [k,
                                                          k+sum(self.compOrder),
                                                          k+sum(self.compOrder)+1]
                                                self.ARorder = k
                                                self.tau = [tau0, tau1, tau2]
                                                self.delays = [d0, d1, d2]
                                                self.numSeq
                                                parameter_hat, _, _, _, _, _ = \
                                                self.cyclic_decent_method(
                                                                 data_train, ch)
                                                data_test["coeff"] = 1
                                                auc_, _,_,_ = self.model_eval(
                                                                parameter_hat,
                                                                data_test, [ch])
                                                parameter, loglike, sigma_hat, \
                                                _, _, _  = \
                                                    self.cyclic_decent_method(
                                                                 data_test, ch)
                                                hyperparameters.append([k,
                                                                        self.tau,
                                                    self.delays, self.compOrder])
                                                error.append(sigma_hat)
                                                auc.append(auc_)
                        # Find the optimal ARorder, tau, delay, and compOrder according to AUC
                        indx1 = [i for i in range(len(auc)) if auc[i] == max(auc)]
                        indx2 = [i for i in range(len(indx1)) if error[i] == min(error)]
                        self.tau = hyperparameters[indx1[indx2]][1]
                        self.delays = hyperparameters[indx1[indx2]][2]
                        self.compOrder = hyperparameters[indx1[indx2]][3]
                        hyperparameters = []
                        hyperparameters.append([k, self.tau, self.delays,
                                                self.compOrder])
                        nParam = [k, k + sum(self.compOrder),
                                  k + sum(self.compOrder) + 1]
                        _, L, _, _, _, _ = self.cyclic_decent_method(data_test, ch)
                        # Compute BIC measure for AR model order selection
                        bic.append(L + k*np.log(numSeq_train*(numSeq_test - k)))

                    indx = [i for i in range(len(bic)) if bic[i] == min(bic)]
                    self.ARorder = AR_range[indx]
                    self.tau = hyperparameters[indx][1]
                    self.delays = hyperparameters[indx][2]
                    self.compOrder = hyperparameters[indx][3]

                nParam = [self.ARorder, self.ARorder + sum(self.compOrder),
                          self.ARorder + sum(self.compOrder) + 1]
                parameter_hat[f,ch], _, _, _, _, _ = \
                    self.cyclic_decent_method(data_train, ch)
                auc_ch[f,ch], acc_ch, score_, trialTargetness_ch = \
                    self.model_eval(parameter_hat[f,ch], data_test, ch)
                score.append(score_)
                trialTargetness.append(trialTargetness_ch)

            data_test["coeff"] = self.multiChanenelCoeff(score, trialTargetness)
            auc[f], acc[f], _, _ = self.model_eval(parameter_hat[f, :],
                                         data_test, self.channels)

            #print("AUC:  | ACC: ".format(auc[f], acc[f]))
            # Set current status
            prog.set_stat(f + 1)
            # Update Progress Bar again
            prog.update()

        # Make the Progress Bar final
        prog.end()
        # Pick best parameters for each channel/brain source
        for ch in range(len(self.channels)):
            best_ind = [i for i in range(nFold) if auc_ch[:, ch] == max(auc_ch[:, ch])]
            parameters = parameter_hat[best_ind, ch]

        return auc, acc, parameters

    def data_for_corssValidation(self, data, nFold = 10):
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
        foldSampSize = np.floor(self.numSeq/nFold)
        eeg = data["timeseries"]
        trigOnsets = data["stimOnset"]
        targetOnsets = data["targetOnset"]
        data_train = dict()
        data_test = dict()
        # shuffling data set
        indx = np.random.permutation(self.numSeq)
        eeg_sh = eeg[:,:,indx]
        trigOnsets_sh = trigOnsets[:,indx]
        targetOnsets_sh = targetOnsets[:,indx]
        y_train = []
        y_test = []
        us_train = []
        us_test = []
        ue_train = []
        ue_test = []

        if self.paradigm == "FRP":
            ind_frp_neg = [i for i in range(self.numSeq) if targetOnsets_sh[i] > 0]
            ind_frp_pos = [i for i in range(self.numSeq) if targetOnsets_sh[i] == 0]
            y_train_pos = eeg_sh[:,:,ind_frp_pos]
            y_train_neg = eeg_sh[:,:,ind_frp_neg]
            us_train_pos = trigOnsets_sh[:,ind_frp_pos]
            us_train_neg = trigOnsets_sh[:,ind_frp_neg]
            ue_train_pos = targetOnsets_sh[ind_frp_pos]
            ue_train_neg = targetOnsets_sh[ind_frp_neg]
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
                    y_train.append(np.concatenate((y_train_pos,y_train_neg),2))
                    us_train.append(np.concatenate((us_train_pos,us_train_neg),1))
                    ue_train.append(np.concatenate((ue_train_pos,ue_train_neg),0))
                    y_test.append(np.concatenate((y_test_pos,y_test_neg),2))
                    us_test.append(np.concatenate((us_test_pos,us_test_neg),1))
                    ue_test.append(np.concatenate((ue_test_pos,ue_test_neg),0))
                else:
                    y_test.append(y_train_)
                    us_test.append(us_train_)
                    ue_test.append(ue_train_)
            else:
                if self.paradigm == "FRP":
                    testIndx = range(int(f*np.floor(foldSampSize/2)),
                                     int((f+1)*np.floor(foldSampSize/2)+1))
                    testIndx_pos = min(len(ind_frp_pos), len(testIndx))
                    testIndx_neg = min(len(ind_frp_neg), len(testIndx))
                    y_test_pos = copy(y_train_pos[:,:,testIndx_pos])
                    y_test_neg = copy(y_train_neg[:,:,testIndx_neg])
                    us_test_pos = copy(us_train_pos[:,testIndx_pos])
                    us_test_neg = copy(us_train_neg[:,testIndx_neg])
                    ue_test_pos = copy(ue_train_pos[testIndx_pos])
                    ue_test_neg = copy(ue_train_neg[testIndx_neg])
                    y_test.append(np.concatenate((y_test_pos,y_test_neg),2))
                    us_test.append(np.concatenate((us_test_pos,us_test_neg),1))
                    ue_test.append(np.concatenate((ue_test_pos,ue_test_neg),0))
                    tmp = copy(y_train_pos)
                    y_train_pos = np.delete(tmp, testIndx_pos, 2)
                    tmp = copy(y_train_neg)
                    y_train_neg = np.delete(tmp, testIndx_neg, 2)
                    tmp = copy(us_train_pos)
                    us_train_pos = np.delete(tmp, testIndx_pos, 1)
                    tmp = copy(us_train_neg)
                    us_train_neg = np.delete(tmp, testIndx_neg, 1)
                    tmp = copy(ue_train_pos)
                    ue_train_pos = np.delete(tmp, testIndx_pos, 0)
                    tmp = copy(ue_train_neg)
                    ue_train_neg = np.delete(tmp, testIndx_neg, 0)
                    y_train.append(np.concatenate((y_train_pos,y_train_neg),2))
                    us_train.append(np.concatenate((us_train_pos,us_train_neg),1))
                    ue_train.append(np.concatenate((ue_train_pos,ue_train_neg),0))
                else:
                    testIndx = range(int(f*np.floor(foldSampSize)),
                                     int((f+1)*np.floor(foldSampSize)+1))
                    y_test.append(y_train_[:,:,testIndx])
                    us_test.append(us_train_[:,testIndx])
                    ue_test.append(ue_train_[0,testIndx])
                    tmp = copy(y_train_)
                    y_train.append(np.delete(tmp, testIndx, 2))
                    tmp = copy(us_train_)
                    us_train.append(np.delete(tmp, testIndx, 1))
                    tmp = copy(ue_train_)
                    ue_train.append(np.delete(tmp, testIndx, 1))

        return y_test, y_train, us_test, us_train, ue_test, ue_train


    def model_eval(self, parameter, data, channel):
        """
        with the final belief over the system, updates the querying method and
        generates len_query most likely queries.
            Input Args:
                parameters: model parameters for each channel/brain sources
                data: a dictionary including timeseries, onsets of stimuli,
                    and targets for all sequences
                channel: set of channels/brain sources (int)
            Return:
                auc: AUC of the classifier
                acc: accuracy of the classifier
                scores: loglikelihood scores of of all trials
                trialTargetness: represents targetness of each trial across all sequences
        """
        numSeq = data["numSeq"]
        y = data["timeseries"]
        us = data["stimOnset"]
        ue = data["targetOnset"]
        # Initialization
        label = np.zeros((numSeq,1))
        coeff = data["coeff"]
        sc = np.zeros((self.numTrial, numSeq))
        score = np.zeros((len(channel), numSeq*(self.numTrial+1)))
        if self.paradigm == "FRP":
            loglikelihood = np.zeros((numSeq, 2, len(channel)))
            trialTargetness = np.zeros((numSeq, 2))
        else:
            loglikelihood = np.zeros((numSeq, self.numTrial, len(channel)))
            trialTargetness = np.zeros((numSeq, self.numTrial))
        # Computing loglikehood scores for each possible target location
        for seq in range(numSeq):
            if self.paradigm == "FRP":
                if us[seq] > 0:
                    label[seq] = 1
                    trialTargetness[seq,1] = 1
                    trialTargetness[seq,0] = 0
                else:
                    label[seq] = 0
                    trialTargetness[seq,1] = 0
                    trialTargetness[seq,0] = 1

                loglikelihood[seq,0,:] = self.loglikelihoodARX(y[channel,:,seq],
                                                      us[:,seq], 0, parameter)
                loglikelihood[seq,1,:] = self.loglikelihoodARX(y[channel,:,seq],
                                                      us[:,seq], us[:,seq],
                                                               parameter)
            else:
                targetLoc = ue[seq]
                possibleTarget = np.concatenate(([0], us[:,seq]))
                for trial in range(self.numTrial):
                    if np.abs(possibleTarget[trial] - targetLoc) < 3:
                        label[seq] = trial + 1
                        trialTargetness[seq,trial] = 1
                    else:
                        trialTargetness[seq,trial] = 0
                    loglikelihood[seq,trial,:] = self.loglikelihoodARX(
                                             y[channel,:,seq], us[:,seq],
                                            possibleTarget[trial], parameter)
        # Computing auc and acc of the classifier
        if self.paradigm == "FRP":
            for ch in range(len(channel)):
                sc[ch,:] = np.matmul(np.array([[-1,1]]), (-1)*loglikelihood[:,:,ch].T)

            # combining multi-channels/brain sources
            scores = np.matmul(coeff.T, sc)
            auc = self.calculateAUC(scores, label)
            acc = self.calculateACC(scores, label)
            trialTargetness = label

        else:
            for ch in range(len(channel)):
                ss = (-1)*loglikelihood[:,:,ch]
                for seq in range(numSeq):
                    sc[:, seq] = ss[seq,:] - ss[seq, trialTargetness[seq,:] > 0]
                    sc[sc[:, seq] ==0, seq] = (-1)*np.mean(sc[sc[:, seq]!=0, seq])

                score[ch,:] = np.reshape(sc,1,(self.numTrial+1)*numSeq)
            scores = np.matmul(coeff.T, score)
            t_target = np.reshape(trialTargetness,1,(self.numTrial+1)*numSeq)
            auc = self.calculateAUC(scores, t_target)
            acc = self.calculateACC(scores, t_target)
            trialTargetness = []
            trialTargetness = t_target

        return auc, acc, scores, trialTargetness

    def cyclic_decent_method(self, data, channel):
        """
        The cyclic descent algorithm is used for joint estimation f the AR(p) and design matrix coefficients.
        For more information check out the following papers:
            - W. Q. Malik, et al., Denoising two-photon calcium imaging data, PloS one, 2011.
            - Y. M. Marghi, et al., A Parametric EEG Signal Model for BCIs with Rapid-Trial Sequences, EMBC, 2018.

            Input Args:
                data: a dictionary including timeseries, onsets of stimuli,
                    and targets for all sequences
                channel: set of channels/brain sources (int)
            Return:
                parameter: estimated parameters fpr the ARX model including AR parameters, gammafunction

                loglikelihood: loglikelihood scores
                sigma_hat: estimation error
                y_hat: synthetic eeg signal
                signal_hat: estimated brain activities due to visual stimuli including VEP+ERP
                s_hat: estimated AR process in the signal model
        """
        # Initialization
        eeg = data["timeseries"][channel,:,:]
        trigOnsets = data["stimOnset"]
        numSeq = data["numSeq"]
        N = self.numSamp - self.ARorder
        Ix = np.identity(N)
        err, b = self.arburg(eeg[:, 0])
        var_hat = .1 * err
        var_hat_old = 1.5 * err
        Q_inv = copy(Ix)
        Q_inv_nonFRP = copy(Ix)
        Q_inv_FRP = copy(Ix)
        iter = 0
        parameter = []
        s_hat = np.zeros((self.numSamp, numSeq))
        X = []  # design matrix
        # Parameter estimation using cyclic decent method
        while abs(var_hat - var_hat_old) > self.threshold * abs(var_hat_old):
            iter += 1
            var_hat_old = var_hat
            sig_hat = np.zeros((self.numSamp, numSeq))
            ar_hat = np.zeros((self.numSamp, numSeq))
            X = []
            y = []
            y_neg = []
            y_pos = []
            arProcess_hat = []
            Q_inv_s = []
            n_neg = 0
            n_pos = 0
            z = np.zeros((np.sum(self.compOrder), self.numSamp - self.ARorder))
            ar_hat = np.zeros((self.numSamp, numSeq))
            if self.paradigm == "FRP":
                beta_s = np.zeros((np.sum(self.compOrder), 2))
            else:
                beta_s = np.zeros((np.sum(self.compOrder), 1))

            for s in range(numSeq):
                vep = np.zeros((self.compOrder[0], N))
                for trial in range(self.numTrial):
                    input = trigOnsets[trial, s] + self.delays[0]
                    vep += self.gammafunction(input, self.compOrder[0], self.tau[0])

                z[0:self.compOrder[0], :] = vep
                if self.paradigm == "FRP":
                    input_p = trigOnsets[0, s] + self.delays[1]
                    input_q = trigOnsets[0, s] + self.delays[2]
                    z[self.compOrder[0]:np.sum(self.compOrder[:2]), :] = \
                        self.gammafunction(input_p, self.compOrder[1], self.tau[1])
                    z[np.sum(self.compOrder[:2]):np.sum(self.compOrder), :] = \
                        self.gammafunction(input_q, self.compOrder[2], self.tau[2])
                elif self.paradigm == "ERP":
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
                    z[np.sum(self.compOrder[:2]):np.sum(self.compOrder), :] = tmp2
                else:
                    warnings.warn(
                        "Please enter a valid paradigm e.g., FRP or ERP!")
                    break

                # updates the design matrix according to the target/non-target stimuli onsets
                X.append(z.T)
                y_s = eeg[self.ARorder:, s]

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
                        sig_hat[self.ARorder:, s] = np.matmul(z.T, beta_s[:, 1])
                    else:
                        # positive sequences
                        y_pos.append(y_s)
                        temp = np.matmul(z, Q_inv)
                        beta_s[:, 0] = np.matmul(
                            np.linalg.inv(np.matmul(temp, z.T)),
                            np.matmul(temp, y_s))
                        # estimated VEP & ERP terms
                        sig_hat[self.ARorder:, s] = np.matmul(z.T, beta_s[:, 0])

                if self.paradigm == "ERP":
                    temp = np.matmul(z, Q_inv)
                    beta_s = np.matmul(np.linalg.inv(np.matmul(temp, z.T)),
                                       np.matmul(temp, y_s))
                    # estimated VEP & ERP terms
                    sig_hat[self.ARorder:, s] = np.matmul(z.T, beta_s)
                    y.append(y_s)

                # AR series - is this ar_hat + res_hat
                arProcess_hat.append(y_s - sig_hat[self.ARorder:, s])
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
                    if data["targetOnset"][s] > 0:
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
            parameter = np.concatenate((alpha_hat, beta_hat_pos, sigma_hat))
            parameter = np.concatenate((alpha_hat, beta_hat_neg, sigma_hat))

        if self.paradigm == "ERP":
            parameter = np.concatenate((alpha_hat, beta_hat, sigma_hat))

        # Compute the loglikelihood
        logdG = np.sum(np.log(D)) + N * np.log(sigma_hat)
        loglikelihood = numSeq * logdG + error / sigma_hat
        # Estimated EEG sequences based on the estimated parameters
        for s in range(numSeq):
            s_tilde = eeg[:self.ARorder, s]
            noise = np.sqrt(sigma_hat) * np.random.randn(self.numSamp)
            s_hat[:self.ARorder, s] = noise[0, :self.ARorder]
            for n in range(self.ARorder + 1, self.numSamp):
                s_hat[n, s] = np.sum(np.flip(s_tilde, 0) * alpha_hat.T) + noise[
                    0, n]
                s_tilde = np.concatenate(
                    (s_tilde[1:], np.expand_dims(s_hat[n, s], 0)))

        signal_hat = np.reshape(sig_hat, (self.numSamp, numSeq))
        y_hat = signal_hat + s_hat

        return parameter, loglikelihood, sigma_hat, y_hat, signal_hat, s_hat

    def gammafunction(self, x, compOrder, tau):
        """
        Generates the brain impulse responses to the visual and target-related stimuli. Here it is assumed that
        this response can be modeled by the gamma function  as follow:

                f(n,tau,L) = sum_l(b_l*(n/l)^l*exp(-n/tau)) u[n]

        for more information, please check equation (4) in "A Parametric EEG Signal Model for BCIs with Rapid-Trial
        Sequences" paper.
            Input Args:
                x: is Nx1
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
        # x = np.array(x)
        try:
            N = x.shape[0]
            Onset0 = x[0]
        except:
            N = 1
            Onset0 = x

        for s in range(N):
            M = self.numSamp - Onset0 - s * dn + 1
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

        temp = np.zeros((compOrder, int(Onset0 - self.ARorder - 1)))
        y_gamma = np.concatenate((temp, y), 1)

        return y_gamma

    def invCov(self, x, L_inv, w_inv):
        """
        Calculate inverse of AR covariance matrix via LDR V is NXps ensemble of  residuals, Linv & Winv are
        initialized to I_N
        """
        # Find AR coefficients
        err, b = self.arburg_(x)
        L_invt = copy(L_inv)
        N = self.numSamp - self.ARorder
        # Transpose of L_invt matrix as per Levinson-Durbin recursion
        for ar in range(self.ARorder):
            coef = b[ar]
            L_invt[:ar + 1, ar + 1] = coef

        for ar in range(N - self.ARorder - 1):
            try:
                L_invt[ar + 1:ar + self.ARorder + 1, ar + self.ARorder] = coef
            except:
                tt = 1

        # Elements of diagonal matrix as per Levinson-Durbin recursion
        x_xcov = np.cov(x, bias=False, ddof=0)
        d_vec = np.concatenate((np.array([[x_xcov]]), err), 0)
        d_inv = float(1) / d_vec
        q = err[-1]
        q_ = float(1) / q
        # Vectorized inverse covariance of AR(p) - Cholesky Form
        A = L_invt[:self.ARorder + 1, :self.ARorder + 1]
        B = L_invt[:self.ARorder + 1, self.ARorder + 1:self.numSamp]
        C = L_invt[self.ARorder + 1:self.numSamp, self.ARorder + 1:self.numSamp]
        D = np.diag(d_inv[:, 0])
        AD = np.matmul(A, D)
        w_inv = copy(w_inv)
        w_inv[:self.ARorder + 1, :self.ARorder + 1] = \
            np.matmul(AD, A.T) + q_ * np.matmul(B, B.T)
        w_inv[:self.ARorder + 1,
        self.ARorder + 1:self.numSamp] = q_ * np.matmul(B, C.T)
        w_inv[self.ARorder + 1:self.numSamp,
        :self.ARorder + 1] = q_ * np.matmul(C, B.T)
        w_inv[self.ARorder + 1:self.numSamp,
        self.ARorder + 1:self.numSamp] = q_ * np.matmul(C, C.T)

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
        N = self.numSamp - self.ARorder
        Ix = np.identity(N)
        z = np.zeros((np.sum(self.compOrder), self.numSamp - self.ARorder))
        ar_hat = np.zeros((self.numSamp, 1))
        log_score = np.zeros((y.shape[0], 1))
        nParam = [self.ARorder, self.ARorder + sum(self.compOrder),
                          self.ARorder + sum(self.compOrder) + 1]
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
            if ue > 0:
                ind = 2
            else:
                ind = 1
            # compute scores for each channel
            for ch in range(y.shape[0]):
                beta = parameter[nParam[0]:nParam[1], ch]
                sigma_hat = parameter[-1, ch]
                # estimated VEP & ERP terms
                sig_hat = np.matmul(z.T, beta[:, 1])
                # AR series - is this ar_hat + res_hat
                arProcess_hat = y[self.ARorder:, 1] - sig_hat
                Q_inv, D, _ = self.invCov(arProcess_hat, Ix, Ix)
                error = np.matmul(arProcess_hat.T, np.matmul(Q_inv,
                                                              arProcess_hat))
                logdG = np.sum(np.log(D)) + N * np.log(sigma_hat)
                log_score[ch] = logdG + error / sigma_hat

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
                arProcess_hat = y[ch, self.ARorder:] - sig_hat
                Q_inv, D, _ = self.invCov(arProcess_hat, Ix, Ix)
                error = np.matmul(arProcess_hat.T, np.matmul(Q_inv,
                                                              arProcess_hat))
                logdG = np.sum(np.log(D)) + N * np.log(sigma_hat)
                log_score[ch] = logdG + error / sigma_hat

        return log_score

    def arburg(self, x):
        """
        AR parameter estimation via Burg method.
        A = ARBURG(X,ORDER) returns the coefficients of the AR parametric signal model estimate
        of X using Burg's method. The model has order ORDER, and the output array A has ORDER+1 columns.
        The coefficients along the Nth row of A model the Nth column of X.  If X is a vector then A is
        a row vector.

        [A,E] = ARBURG(...) returns the final prediction error E (the variance
        estimate of the white noise input to the AR model).

        [A,E,K] = ARBURG(...) returns the reflection coefficients (parcor
        coefficients) in each column of K.
            Input Args:
                x: is Nx1
            Return:
                err: final prediction error (estimated white noise variance)
                k: has size 1xm and m can range from 1 to AR order, error is mXps
        """

        # Initialization
        N = float(x.shape[0])
        a = [[1]]
        err = np.sum(x * x) / N
        k = np.zeros((self.ARorder, 1))
        efp = x[1:]
        ebp = x[:-1]
        # Burg algorithm recursions
        for ar in range(self.ARorder):
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
            # coef = float(-1) * a[0][1:]
            # coef = float(-1) * np.flip([coef], 1)
            # coef = np.concatenate(([1],coef[0]))
            # Update the prediction error
            err = (1 - kk * kk) * err

        return err, a

    def arburg_(self, x):
        """
        This function Vectorized AR parameter estimation via Burg method
            Input Args:
                x: is Nx1
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
        err = np.zeros((self.ARorder + 1, 1))
        err[0] = np.sum(x * x) / N
        k = np.zeros((self.ARorder, 1))

        # Burg algorithm recursions
        for ar in range(self.ARorder):
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

    def calculateAUC(self, scores, labels):
        """
        Computes the area under the ROC curve (AUC) of a binary classifier.
            Input Args:
                scores: classifier scores
                labels: true labels
            Return:
                auc: the area under the ROC curve
        """
        fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        return auc

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
        testLabels = []
        predict = []
        numSamp = scores.shape[0]
        # Predicting labels according to the scores
        for s in range(numSamp):
            sc = scores[s,:]
            indx = [i for i in range(self.numTrial) if sc[i] == min(sc)]
            sc = np.zeros((1,self.numTrial))
            sc[indx] = 1
            predict.append(sc)
            testLabels.append(labels[s,:])

        acc = 1 - np.sum(np.abs(predict - testLabels))/(numSamp*self.numTrial)

        return acc

