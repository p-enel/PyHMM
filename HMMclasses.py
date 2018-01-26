import numpy as np
import functools as fct
import pickle
# import warnings

# warnings.simplefilter("error")


class NumericalInstability(Exception):
    pass


def vecfac(x):
    return np.array([np.math.factorial(x[i]) for i in range(len(x))])[:, None]


def logdot(x, y):
    max_x, max_y = np.max(x, 1, keepdims=True), np.max(y, 0, keepdims=True)
    exp_x, exp_y = x - max_x, y - max_y
    np.exp(exp_x, out=exp_x)
    np.exp(exp_y, out=exp_y)
    exp_x[np.isnan(exp_x)] = 0
    exp_y[np.isnan(exp_y)] = 0
    c = np.dot(exp_x, exp_y)
    np.log(c, out=c)
    c += max_x + max_y
    return c


def logvecsum(a, dim=0, keepdims=False):
    max_a = np.max(a, dim, keepdims=True)
    exp_a = a - max_a
    if not keepdims:
        max_a = np.squeeze(max_a, dim)
    np.exp(exp_a, out=exp_a)
    s = np.sum(exp_a, dim, keepdims=keepdims)
    return np.log(s) + max_a


class HMM_base(object):
    """The base class for different types of HMM algorithm
    """

    def __init__(self):
        self.nstates = None
        self.pi = None
        self.A = None
        self.B = None
        self.nfeatures = None
        self._hasparameters = False
        self._PO = None

    def set_params(self, A, B, pi):
        '''Set the parameters of the HMM by hand

        Arguments:
        A -- np.array<nstates*nstates> - the transition probability matrix
        B -- np.array<nfeatures*nstates> - the emission probability matrix
        pi -- np.array<nstates> - the initial state probability vector
        '''
        try:
            assert np.all([isinstance(a, np.ndarray) for a in [A, B, pi]])
            assert np.ndim(A) == 2 and np.ndim(B) == 2 and np.ndim(pi) == 1
        except AssertionError:
            errmsg = "Arguments must be np.arrays with dimension 2 for A and B"
            errmsg += " and dimension 1 for pi"
            raise ValueError(errmsg)
        if A.shape[0] != A.shape[1]:
            raise ValueError(
                "The transition matrix (1st arg) must be a square matrix")
        if A.shape[0] != B.shape[1]:
            raise ValueError(
                "Transition and emission matrices are not consistent")
        if A.shape[0] != len(pi):
            errmsg = "Transition matrix and initial prob vector are not "
            errmsg += "consistent"
            raise ValueError(errmsg)
        self.A = A
        self.B = B
        self.pi = pi.reshape(1, -1)
        self.nstates = A.shape[0]
        self.nfeatures = B.shape[0]
        self._hasparameters = True
        self._POs = None

    @staticmethod
    def _check_obserbations(observations):
        """Check that the observations follow the appropriate format

        Arguments:
        observations -- [nsequences] np.array<nemissions*nfeatures> - the list of sequences of emissions
        """
        errmsg = "The argument of observation sequence must either be a "
        errmsg += "list of np.array or a single np.array"
        if not isinstance(observations, (list, np.ndarray)):
            raise ValueError(errmsg)

        if isinstance(observations, list):
            if not np.all([isinstance(seq, np.ndarray) for seq in
                           observations]):
                raise ValueError(errmsg)

    def evaluate(self, observations):
        """Return the log of the probability of observing the argument observation sequence given the parameters of the model, i.e. the log likelihood of the model

        Arguments:
        observations -- [nsequences] np.array<nemissions*nfeatures> - the list of sequences of emissions

        Returns:
        logPO -- float - the log likelihood of the model
        """
        if not self._hasparameters:
            print("You must set or estimate parameters first")
            raise ValueError
        self._check_obserbations(observations)
        if isinstance(observations, np.ndarray):
            observations = [observations]

        logPO = 0
        for seq in observations:
            emsprob = self._ems_prob(seq)
            logprob, _, _ = self._forward(seq, emsprob)
            logPO += logprob
        return logPO

    def state_posteriors(self, observations):
        """The posterior probabilities of being in a state given the argument observation sequence and the model parameters

        Arguments:
        observations -- [nsequences] np.array<nemissions*nfeatures> - the list of sequences of emissions

        Returns:
        posteriors -- [nsequences] np.array<nemissions*nstates> - the posterior probability of each states for each emission in all the sequences
        """
        if not self._hasparameters:
            print("You must set or estimate parameters first")
            return None
        self._check_obserbations(observations)
        if isinstance(observations, np.ndarray):
            observations = [observations]
        posteriors = []
        for seq in observations:
            emsprob = self._ems_prob(seq)
            _, fwdlattice, coeffs = self._forward(seq, emsprob)
            bwdlattice = self._backward(seq, emsprob, coeffs)
            prod = fwdlattice * bwdlattice
            posterior = prod / np.sum(prod, 1, keepdims=True)
            posteriors.append(posterior)
        return posteriors

    def save_parameters(self, filename='savedHMM.pk'):
        """Save parameters of the model into a file

        Arguments:
        filename -- str - the path of the file to pickle the parameters in
        """
        with open(filename, 'wb') as f:
            pickle.dump([self.A, self.B, self.pi], f)
        print("Parameters saved in file:", filename)

    def load_parameters(self, filename='savedHMM.pk'):
        """Load saved parameters into the model

        Arguments:
        filename -- str - the path of the pickle file containing the parameters to load
        """
        with open(filename, 'rb') as f:
            self.A, self.B, self.pi = pickle.load(f)
        self.nstates = self.A.shape[0]
        self.nfeatures = self.B.shape[0]
        print("Parameters loaded from file:", filename)
        self._hasparameters = True

    def _ems_prob(self, sequence):
        """Calculate the probability of each emission in a sequence given the transition probabilities
        Arguments:
        sequence -- <np.array nemissions*nfeatures> a sequence of emissions
        """
        # Overriden in subclass
        raise NotImplementedError

    def _forward(self, sequence, emsprob):
        """Calculate the forward variable

        Arguments:
        sequence -- np.array<nemissions*nfeatures> - a sequence of emissions
        emsprob -- np.array<nemissions*nstates> - the probability of each emission to have been emitted by each state

        Returns:
        logprob -- float - the log probability of a sequence of emissions
        fwdlattice -- np.array<nemissions*nstates> - the forward lattice
        coeffs -- np.array<nemissions> - the normalization coefficients used to prevent underflow
        """
        # Calculate the probabilities of each emssion
        T = len(sequence)
        fwdlattice = np.empty((T, self.nstates))
        coeffs = np.zeros(T)
        fwdlattice[0, :] = self.pi * emsprob[0, :]
        coeffs[0] = 1 / np.sum(fwdlattice[0, :])
        fwdlattice[0, :] *= coeffs[0]
        for t in range(1, T):
            fwdlattice[t, :] = np.dot(
                fwdlattice[t - 1, :], self.A) * emsprob[t, :]
            coeffs[t] = 1 / np.sum(fwdlattice[t, :])
            fwdlattice[t, :] *= coeffs[t]

        logprob = -np.sum(np.log(coeffs))
        return logprob, fwdlattice, coeffs

    def _backward(self, sequence, emsprob, coeffs):
        """Calculate the backward variable

        Arguments:
        sequence -- np.array<nemissions*nfeatures> a sequence of emission
        coeffs -- np.array<nemissions> the scaling coefficients obtained with the forward variable
        emsprob -- np.array<nemissions*nstates> the probability of each emission to have been emitted by each state

        Returns:
        bwdlattice -- np.array<nemissions*nstates> - the backward lattice
        """
        T = len(sequence)
        bwdlattice = np.empty((T, self.nstates))
        bwdlattice[-1, :] = np.ones(self.nstates) * coeffs[-1]
        for t in range(0, T - 1)[::-1]:
            bwdlattice[t, :] = np.dot(self.A * emsprob[t + 1, :][None, :],
                                      bwdlattice[t + 1, :].T).T * coeffs[t]
        return bwdlattice

    def decode(self, observations):
        """Use the Viterbi algorithm to find the most likely sequence of states given the arguments observation sequence and the parameters of the model

        Arguments:
        observations -- [nsequences] np.array<nemissions*nfeatures> a list of sequences of emissions

        Returns:
        maxpath -- [nsequences] np.array<nemissions> - the most likely state sequence for each emission sequence
        maxlogprob -- [nsequences] float - the probability of the most likely state sequence for each emission sequence
        """
        self._check_obserbations(observations)
        if isinstance(observations, np.ndarray):
            observations = [observations]
        maxpath = []
        maxlogprob = []
        self._logA = np.log(self.A)
        self._logB = np.log(self.B)
        self._logpi = np.log(self.pi)
        for seq in observations:
            logemsprob = np.log(self._ems_prob(seq))
            path, prob = self._viterbi(seq, logemsprob)
            maxpath.append(path)
            maxlogprob.append(prob)
        return maxpath, maxlogprob

    def _viterbi(self, sequence, logemsprob):
        """The Viterbi algorithm for a single sequence

        Arguments:
        sequence -- np.array<nemissions*nfeatures> - a single sequence of emissions
        logemsprob -- np.array<nemissions*> - the logarithm of the the emission probabilitis

        Returns:
        maxpath -- np.array<nemissions> - the most likely state sequence
        maxlogprob -- float - the probability of the most likely state sequence
        """
        delta = self._logpi + logemsprob[0, :]
        T = len(sequence)
        psi = np.zeros((T, self.nstates), dtype=np.int32)
        for t in range(1, T):
            ems = sequence[t]
            d_a = delta.T + self._logA
            delta = np.max(d_a, 0, keepdims=True) + logemsprob[ems]
            psi[t, :] = np.argmax(d_a, 0)
        maxlogprob = np.max(delta)
        maxpath = np.empty(T, dtype=int)
        maxpath[-1] = np.argmax(delta)
        for t in range(1, T)[::-1]:
            maxpath[t - 1] = psi[t, maxpath[t]]

        return maxpath, maxlogprob

    def _print_estimate(self, i, logPO, logPOdiff, printwidth, end):
        print('{0:{width}{base}}'.format(
            i, base='d', width=printwidth), end=end)
        print('{0:{width}{base}}'.format(
            logPO, base='.1f', width=printwidth), end=end)
        if logPOdiff > 1:
            base = '.2f'
        else:
            base = '.2e'
        print('{0:{width}{base}}'.format(
            logPOdiff, base=base, width=printwidth - 5), end=end)
        print()

    def estimate(self, observations, niterations=1e10, tol=1e-6, verbose=True):
        """Estimate the parameters that maximize the log likelihood of the model given the data with the iterative Baum-Welch algorithm

        Arguments:
        observations -- [nsequences] np.array<nemissions*nfeatures> - a list of sequences of emissions
        niterations -- int - the maximum number of iterations
        tol -- float - the tolerance of log likelihood (ll) difference between iterations. If the difference in ll between two iterations is lower than this number the algorithm stops
        verbose -- boolean - print progression of the algorithm

        Returns:
        logPOs -- [niterations] float - the log likelihood of the model at each iteration
        """
        self._check_obserbations(observations)
        if isinstance(observations, np.ndarray):
            observations = [observations]
        initlogPO = self.evaluate(observations)
        if verbose:
            print("Initial log likelihood = %.3f" % initlogPO)
            print("Starting Baum-Welch algorithm with criterions:")
            if niterations:
                print("Max number of iterations: %d" % niterations)
            else:
                niterations = 9999999999
            print("Min difference in likelihood: %.1e" % tol)
            print()

            head = ["Iteration #", "log likelihood", "ll diff"]
            printwidth = 15
            end = '    '
        i = 0
        logPOs = []
        while True:
            newA, newB, newpi, logPO = self._EM_iter(observations)
            self.A = newA
            self.B = newB
            self.pi = newpi
            if i != 0:
                logPOdiff = logPO - logPOs[-1]
            else:
                logPOdiff = 9999999999
            logPOs.append(logPO)

            if verbose:
                if i % 25 == 0:
                    print()
                    for title in head:
                        print('{0:{width}{base}}'.format(
                            title, base='s', width=printwidth), end=end)
                    print()
                if i != 0:
                    self._print_estimate(i, logPO, logPOdiff, printwidth, end)

            i += 1
            if i >= niterations or logPOdiff < tol:
                break
        logPO = self.evaluate(observations)
        logPOdiff = logPO - logPOs[-1]
        logPOs.append(logPO)
        if verbose:
            self._print_estimate(i, logPO, logPOdiff, printwidth, end)
        return logPOs

    def _EM_iter(self, observations):
        """The core of the Baum-Welch algorithm

        Arguments:
        observations -- [nsequences] np.array<nemissions*nfeatures> - a list of sequences of emissions

        Returns:
        newA -- np.array<nstates*nstates> - the new transition matrix
        newB -- np.array<nfeatures*nstates> - the new emission matrix
        newpi -- np.array<1*nstates> - the initial probabilities
        """
        logPO = 0
        newpi = np.zeros_like(self.pi)
        newA = np.zeros_like(self.A)
        newB = np.zeros_like(self.B)
        newBdenom = np.zeros_like(self.pi)
        for iseq, seq in enumerate(observations):
            emsprob = self._ems_prob(seq)
            seqlogPO, fwdlattice, coeffs = self._forward(seq, emsprob)
            logPO += seqlogPO
            bwdlattice = self._backward(seq, emsprob, coeffs)
            xi_sum, gamma_per_feature, gamma_sum, piseq = self._estimate_seq(
                seq, fwdlattice, bwdlattice, emsprob, coeffs)
            newA += xi_sum
            newB += gamma_per_feature
            newBdenom += gamma_sum
            newpi += piseq
        newA /= np.sum(newA, 1, keepdims=True)
        newB /= newBdenom
        newpi /= np.sum(newpi)

        return newA, newB, newpi, logPO

    def _estimate_B_seq(self, sequence, loggamma):
        """Calculate the numerator of the B estimates for a single sequence

        Arguments:
        sequence -- <np.array nemissions*nfeatures> - a sequence of emissions
        loggamma -- <np.array nemissions*nstates> - the logarithm of the gammas for each time step
        """
        raise NotImplementedError

    def _estimate_seq(self, sequence, fwdlattice, bwdlattice, emsprob, coeffs):
        """Part of the Baum-Welch algorithm for a single sequence
        See Rabiner 1989 to understand the name of the variables xi and gamma

        Arguments:
        sequence -- np.array<nemissions*nfeatures> - a sequence of emission
        fwdlattice -- np.array<nemissions*nstates> - the forward variable lattice
        bwdlattice -- np.array<nemissions*nstates> - the backward variable lattice
        coeffs -- np.array<nemissions> - the scaling coefficients obtained with the forward variable

        Returns:
        xisum -- np.array<nstates*nstates*1> - the sum of xi for the current sequence
        gammaperfeat -- np.array<nfeatures*nstates> - the sums of gammas for each feature
        gammasum -- np.array<nstates> - the sum of all gammas for this sequence
        piseq -- np.array<1*nstates> - the probability of states for the first emission in this sequence
        """
        logfwd = np.log(fwdlattice)
        logbwd = np.log(bwdlattice)

        # Estimation of transition matrix
        fwdtmp = np.moveaxis(logfwd[:, :, None], 0, 2)
        bwdtmp = np.swapaxes(logbwd[:, :, None], 0, 2)
        Btmp = np.log(np.swapaxes(emsprob[:, :, None], 0, 2))

        xi = np.exp(fwdtmp[:, :, :-1] + bwdtmp[:, :, 1:] +
                    np.log(self.A[:, :, None]) + Btmp[:, :, 1:])
        xisum = np.sum(xi, 2)

        # Estimation of emission matrix
        loggamma = np.log(np.sum(xi, 1))
        loggamma = logfwd[:-1, :] + logbwd[:-1, :] - \
            np.log(coeffs[:-1][:, None])

        # Estimation of the initial probabilities
        piseq = np.exp(loggamma[0, :][None, :])
        gammaperfeat, gammasum = self._estimate_B_seq(sequence, loggamma)

        return xisum, gammaperfeat, gammasum, piseq


class HMM_symbols(HMM_base):
    """HMM model for single symbol observation sequences

    """

    def _ems_prob(self, sequence):
        """Calculate the probability of each emission in a sequence given the transition probabilities

        Arguments:
        sequence -- np.array<nemissions*nfeatures> - a sequence of emissions

        Returns the emission probability <np.array nemissions>
        """
        return self.B[sequence]

    def _estimate_B_seq(self, sequence, loggamma):
        """Calculate the numerator of the B estimates for a single sequence

        Arguments:
        sequence -- np.array<nemissions*nfeatures> - a sequence of emissions
        loggamma -- np.array<nemissions*nstates> - the logarithm of the gammas for each time step

        Returns:
        gammapersmb -- np.array<nfeatures*nstates> - the sums of gammas for each feature
        gammasum -- np.array<nstates> - the sum of all gammas for this sequence
        """
        gammapersmb = np.empty_like(self.B)
        for symbol in range(self.nfeatures):
            gammapersmb[symbol, :] = np.sum(np.exp(
                loggamma[sequence[:-1] == symbol, :]), 0)
        gammasum = np.exp(logvecsum(loggamma, 0))
        return gammapersmb, gammasum


class HMM_Poisson(HMM_base):
    """HMM model for multiple poisson observation sequences
    """

    def _estimate_seq(self, sequence, fwdlattice, bwdlattice, emsprob, coeffs):
        """Part of the Baum-Welch algorithm for a single sequence

        Arguments:
        sequence -- np.array<nemissions*nfeatures> - a sequence of emission
        fwdlattice -- np.array<nemissions*nstates> - the forward variable lattice
        bwdlattice -- np.array<nemissions*nstates> - the backward variable lattice
        coeffs -- np.array<nemissions> - the scaling coefficients obtained with the forward variable

        Returns:
        xi_sum --
        gamma_per_feature --
        gamma_sum --
        piseq --
        """

        # Estimation of transition matrix
        fwdtmp = np.moveaxis(fwdlattice[:, :, None], 0, 2)
        bwdtmp = np.swapaxes(bwdlattice[:, :, None], 0, 2)
        Btmp = np.swapaxes(emsprob[:, :, None], 0, 2)

        xi = fwdtmp[:, :, :-1] * bwdtmp[:, :, 1:] * \
            self.A[:, :, None] * Btmp[:, :, 1:]
        xi_sum = np.sum(xi, 2)

        # Estimation of emission matrix
        gamma = np.sum(xi, 1).T
        # gamma = fwdlattice[:-1, :] * bwdlattice[:-1, :] / coeffs[:-1][:, None]
        # posterior = gamma / gamma.sum(1, keepdims=True)
        seq = sequence[:-1].astype(float).T

        gamma_per_feature = np.dot(seq, gamma)
        # gamma_sum = gamma.sum(0)
        gamma_sum = gamma.sum(0)

        # Estimation of the initial probabilities
        # piseq = gamma[0, :][None, :]
        piseq = gamma[0, :][None, :]

        return xi_sum, gamma_per_feature, gamma_sum, piseq

    def _ems_prob(self, sequence):
        """Calculate the probability of each emission in a sequence given the transition probabilities
        Arguments:
        sequence -- <np.array nemissions*nfeatures> a sequence of emissions

        Returns:
        emsprob -- float - the emission probability
        """
        Bhash = self.B.tostring()
        emsprob = np.empty((sequence.shape[0], self.nstates))
        for iems, ems in enumerate(sequence):
            self.ems = ems[:, None]
            # the transpose is easier to read, it has no effect on the computation
            emshash = self.ems.T.tostring()
            emsprob[iems] = self._ems_prob_single(Bhash, emshash)
        return emsprob

    @staticmethod
    def _vec_factorial(x):
        """A vectorized version of factorial"""
        return np.array([np.math.factorial(x[i]) for i in range(len(x))])[:, None]

    @fct.lru_cache(maxsize=2000)
    def _ems_prob_single(self, Bhash, emshash):
        """Getting the likelihood of a single observation with the poisson distribution whith parameter self.B

        Arguments:
        Bhash -- str - B converted into a string which serves as a hash for this function extended with caching
        emshash -- str - an emission converted into a string which serves as a hash for this function extended with caching. The original size of ems is <np.array nfeatures*1>.

        Returns:
        emsprob -- float - emission probability
        """
        # Using log space here to prevent underflow
        tmp = np.exp(-self.B) * self.B**self.ems /\
            self._vec_factorial(self.ems)
        emsprob = np.prod(tmp,
                          axis=0,
                          dtype=float,
                          keepdims=True)
        return emsprob
