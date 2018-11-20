import numpy as np
import warnings


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

    def __init__(self, fs, filename, paradigm, numTrial, numSeq, numSamp,
                 hyperparameter, channels, orderSelection=False,
                 nfold=10, threshold=1e-6):
        self.fs = fs
        self.nfold = nfold
        self.filename = filename
        self.paradigm = paradigm
        self.channels = channels
        self.orderSelection = orderSelection
        self.numSamp = numSamp
        self.numSeq = numSeq
        self.ARorder = hyperparameter[0]
        self.tau = hyperparameter[]
        self.delays = hyperparameter[]
        self.threshold = threshold
        self.numTrial = numTrial

    def sequence_model(self, p):
        """ with the final belief over the system, updates the querying method and
            generates len_query most likely queries.
            Args:
                p(list[float]): list of probability distribution over the state estimates
                len_query(int): number of queries in the scheduled query
            Return:
                query(list[str]): queries """

        return []

    def model_eval(self, p):
        """ with the final belief over the system, updates the querying method and
            generates len_query most likely queries.
            Args:
                p(list[float]): list of probability distribution over the state estimates
                len_query(int): number of queries in the scheduled query
            Return:
                query(list[str]): queries """

        return []

    def cyclic_decent_method(self, data):
        """
        The cyclic descent algorithm is used for joint estimation f the AR(p) and design matrix coefficients.
        For more information check out the following papers:
            - W. Q. Malik, et al., Denoising two-photon calcium imaging data, PloS one, 2011.
            - Y. M. Marghi, et al., A Parametric EEG Signal Model for BCIs with Rapid-Trial Sequences, EMBC, 2018.

            Args:
                p(list[float]): list of probability distribution over the state estimates
                len_query(int): number of queries in the scheduled query
            Return:
                query(list[str]): queries """
        # Initialization
        eeg = data.timeseries
        trigOnsets = data.stimOnsets
        N = self.numSamp - self.ARorder
        Ix = np.identity(N)
        err, b = self.arburg(self, eeg[:, 0])
        err = .1 * err
        err_old = 1.5 * err
        Q_inv = Ix
        Q_inv_nonFRP = Ix
        Q_inv_FRP = Ix
        iter = 0
        X = []  # design matrix
        # Parameter estimation using cyclic decent method
        while abs(err - err_old) > self.threshold * abs(err_old):
            iter += 1
            err_old = err
            sig_hat = np.zeros((self.numSamp, self.numSeq))
            ar_hat = np.zeros((self.numSamp, self.numSeq))
            X = []
            Y = []
            y_neg = []
            y_pos = []
            arProcess_hat = []
            Q_inv_s = []
            n_neg = 0
            n_pos = 0
            z = np.zeros((np.sum(self.compOrder), self.numSamp - self.ARorder))
            ar_hat = np.zeros()
            if self.paradigm == "FRP":
                beta_s = np.zeros()
            else:
                beta_s = np.zeros()

            for s in range(self.numSeq):
                input = data.stimOnset[s] + self.delay[0]
                z[0:self.compOrder[0], :] = self.gammafunction(self, input,
                                                               self.compOrder[
                                                                   0],
                                                               self.tau[0])

                if self.paradigm == "FRP":
                    input_p2 = data.stimOnset[s] + self.delay[1]
                    input_p3 = data.stimOnset[s] + self.delay[2]
                    z[self.compOrder[0]:self.compOrder[1], :] = \
                        self.gammafunction(self, input_p, self.compOrder[1],
                                           self.tau[1])
                    z[self.compOrder[1]:self.compOrder[2],
                    :] = self.gammafunction(self, input_q, self.compOrder[2],
                                            self.tau[2])
                elif self.paradigm == "ERP":
                    input_p = data.targetOnset[s] + self.delay[1]
                    input_q = data.targetOnset[s] + self.delay[2]
                    z[self.compOrder[0]:self.compOrder[1],
                    :] = self.gammafunction(self, input_p, self.compOrder[1],
                                            self.tau[1])
                    z[self.compOrder[1]:self.compOrder[2],
                    :] = self.gammafunction(self, input_q, self.compOrder[2],
                                            self.tau[2])
                else:
                    warnings.warn(
                        "Please enter a valid paradigm e.g., FRP or ERP!")
                    break

                # updates the design matrix according to the target/non-target stimuli onsets
                X.append(z.T)
                y_s = data.eeg[self.ARorder:, s]

                # check the experiment paradigm
                if self.paradigm == "FRP":
                    # separate positive and negative FRP sequences
                    if data.targetOnset[s] > 0:
                        # negative sequences
                        y_neg.append(y_s)
                        temp = np.matmul(z.T, Q_inv)
                        beta_s[:, 1] = np.divide(np.matmul(temp, z),
                                                 np.matmul(temp, y_s))
                        # estimated VEP & ERP terms
                        sig_hat[:, s] = np.matmul(z, beta_s[:, 1])
                    else:
                        # positive sequences
                        y_pos.append(y_s)
                        temp = np.matmul(z.T, Q_inv)
                        beta_s[:, 0] = np.divide(np.matmul(temp, z),
                                                 np.matmul(temp, y_s))
                        # estimated VEP & ERP terms
                        sig_hat[:, s] = np.matmul(z, beta_s[:, 0])

                if self.paradigm == "ERP":
                    temp = np.matmul(z.T, Q_inv)
                    beta_s = np.divide(np.matmul(temp, z), np.matmul(temp, y_s))
                    # estimated VEP & ERP terms
                    sig_hat[:, s] = np.matmul(z, beta_s)
                    y.append(y_s)

                # AR series - is this ar_hat + res_hat
                arProcess_hat.append(y_s - sig_hat[:, s])
                w_inv, d_vec, q = self.invCov(self, arProcess_hat[-1], Ix, Ix)
                Q_inv_s.append(w_inv)

            matX = np.zeros((sum(self.compOrder), sum(self.compOrder)))
            matY = np.zeros((sum(self.compOrder), 1))
            matY_neg = np.zeros((sum(self.compOrder), 1))
            matY_pos = np.zeros((sum(self.compOrder), 1))

            for s in range(self.numSeq):
                err.append(np.matmul(arProcess_hat[0][s].T,
                                     np.matmul(Q_inv, arProcess_hat[0][s])))
                matX += np.matmul(X[0][s].T, np.matmul(Q_inv_s[0][s], X[0][s]))
                # check the experiment paradigm
                if self.paradigm == "FRP":
                    # separate positive and negative FRP sequences
                    if data.targetOnset[s] > 0:
                        # negative sequences
                        matY_neg += np.matmul(X[0][s].T,
                                              np.matmul(Q_inv_s[0][s],
                                                        y_neg[0][s]))
                    else:
                        # positive sequences
                        matY_pos += np.matmul(X[0][s].T,
                                              np.matmul(Q_inv_s[0][s],
                                                        y_pos[0][s]))
                if self.paradigm == "ERP":
                    matY += np.matmul(X[0][s].T,
                                      np.matmul(Q_inv_s[0][s], y[0][s]))

            # Design matrix parameters
            # check the experiment paradigm
            if self.paradigm == "FRP":
                # negative sequences
                beta_hat_neg = np.divide(matX, matY_neg)
                # positive sequences
                beta_hat_pos = np.divide(matX, matY_pos)
            if self.paradigm == "ERP":
                beta_hat = np.divide(matX, matY)

            # Update Q_inv with a detrended V so arburg works well
            Q_inv, d_vec, q = self.invCov(self, arProcess_hat[:], Ix, Ix)
            # Estimate alpha and sigma2e with burg
            sigma_hat, ar_coeff = self.arburg(self, arProcess_hat[:])
            error = np.mean(err)
            # Mean square error - diff. from mse of lscov
            var_hat = sigma_hat

        alpha_hat = (-1)*ar_coeff[1:].T

        return []

    def gammafunction(self, x):
        """
        Generates the brain impulse responses to the visual and target-related stimuli. Here it is assumed that
        this response can be modeled by the gamma function  as follow:

                f(n,tau,L) = sum_l(b_l*(n/l)^l*exp(-n/tau)) u[n]

        for more information, please check equation (4) in "A Parametric EEG Signal Model for BCIs with Rapid-Trial
        Sequences" paper.
            Args:
                x: is Nx1
            Return:
                y_gamma: an array including the value of gamma function
        """
        y = []
        try:
            dn = np.min(np.diff(x))
        except:
            dn = 0
        # x = np.array(x)
        N = x.shape[0]

        for s in range(N):
            M = self.numSamp - x[0] - s * dn + 1
            m = np.array(range(M))
            z = []  # np.zeros((comp_order, M))

            for r in range(self.comp_order):
                z.append(np.power((m / float(r + 1)), (r + 1)) * np.exp(
                    -1. * m / self.tau))

            if dn == 0:
                y += z
            else:
                temp = np.zeros((self.comp_order, s * dn))
                z = z.reshape()
                y += np.concatenate((temp, z), axis=0)

        temp = np.zeros((self.comp_order, x[0] - self.ARorder - 1))
        y_gamma = np.concatenate((temp, y), axis=1)

        return y_gamma

    def invCov(self, x, L_invt, w_inv):
        """
        Calculate inverse of AR covariance matrix via LDR V is NXps ensemble of residuals, Linv & Winv are
        initialized to I_N
            Args:
                x: is Nx1
                L_invt:
                w_inv:
            Return:
                w_inv:
                d_vec:
                q:
        """
        # Find AR coefficients
        err, b = self.arburg_(self, x, L_invt, w_inv)

        # Transpose of L_invt matrix as per Levinson-Durbin recursion
        for ar in range(self.ARorder):
            coef = b[ar]
            L_invt[:ar + 1, ar + 1] = coef

        for ar in range(self.numSamp - self.ARorder - 1):
            L_invt[ar + 1:ar + self.ARorder + 1, ar + self.ARorder + 1] = coef

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
        w_inv[:self.ARorder + 1, :self.ARorder + 1] = np.matmul(AD,
                                                                A.T) + q_ * np.matmul(
            B, B.T)
        w_inv[:self.ARorder + 1,
        self.ARorder + 1:self.numSamp] = q_ * np.matmul(B, C.T)
        w_inv[self.ARorder + 1:self.numSamp,
        :self.ARorder + 1] = q_ * np.matmul(C, B.T)
        w_inv[self.ARorder + 1:self.numSamp,
        self.ARorder + 1:self.numSamp] = q_ * np.matmul(C, C.T)

        return w_inv, d_vec, q

    def loglikelihood(self, p):
        """
        with the final belief over the system, updates the querying method and
        generates len_query most likely queries.
            Args:
                p(list[float]): list of probability distribution over the state estimates
                len_query(int): number of queries in the scheduled query
            Return:
                query(list[str]): queries
        """

        return []

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
            Args:
                x: is Nx1
            Return:
                err: final prediction error (estimated white noise variance)
                k: has size 1xm and m can range from 1 to AR order, error is mXps
        """

        # Initialization
        N = float(self.numSamp)
        b = []
        a = [[1]]
        err = np.zeros((self.ARorder + 1, 1))
        err[0] = np.sum(x * x) / N
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
            coef = float(-1) * a[0][1:]
            coef = float(-1) * np.flip([coef], 1)
            b.append(coef)
            # Update the prediction error
            err = (1 - kk * kk) * err

        return err, b[self.ARorder]

    def arburg_(self, x):
        """
        This function Vectorized AR parameter estimation via Burg method
            Args:
                x: is Nx1
            Return:
                err: final prediction error (estimated white noise variance)
                b{m}: has size 1xm and m can range from 1 to AR order, error is mXps
        """

        # Initialization
        N = float(self.numSamp)
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

    def auc(self, p):
        """ with the final belief over the system, updates the querying method and
            generates len_query most likely queries.
            Args:
                p(list[float]): list of probability distribution over the state estimates
                len_query(int): number of queries in the scheduled query
            Return:
                query(list[str]): queries """

        return []
