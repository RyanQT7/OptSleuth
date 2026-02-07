import numpy as np
import pandas as pd
from math import log, floor
from scipy.optimize import minimize

class SPOT:
    """
    Streaming Peaks-Over-Threshold (SPOT) algorithm for anomaly detection 
    in streaming data.
    Based on Extreme Value Theory (EVT).
    """

    def __init__(self, q=1e-4):
        """
        Args:
            q (float): Detection level (risk probability).
        """
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = 'Streaming Peaks-Over-Threshold Object\n'
        s += f'Detection level q = {self.proba}\n'
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += f'\t initialization : {self.init_data.size} values\n'
            s += f'\t stream : {self.data.size} values\n'
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += f'\t initial threshold : {self.init_threshold}\n'

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += f'\t number of observations : {r} ({100 * r / self.n:.2f} %)\n'
            else:
                s += f'\t number of peaks : {self.Nt}\n'
                s += f'\t extreme quantile : {self.extreme_quantile}\n'
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        """
        Import data to the SPOT object.
        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print(f'This data format ({type(data)}) is not supported')
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) and (init_data < 1) and (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        """
        Add data to the stream.
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print(f'This data format ({type(data)}) is not supported')
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Run the calibration (initialization) phase.
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level

        level = level - floor(level)
        n_init = self.init_data.size

        # We sort X to get the empirical quantile
        S = np.sort(self.init_data)
        # t is fixed for the whole algorithm
        self.init_threshold = S[int(level * n_init)]

        # Initial peaks
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        # Handle case with 0 peaks during initialization
        if self.Nt == 0:
            self.extreme_quantile = self.init_threshold
            return

        if verbose:
            print(f'Initial threshold : {self.init_threshold}')
            print(f'Number of peaks : {self.Nt}')
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print(f'\tGamma = {g}')
            print(f'\tSigma = {s}')
            print(f'\tL = {l}')
            print(f'Extreme quantile (probability = {self.proba}): {self.extreme_quantile}')

        return

    @staticmethod
    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find roots of the given function.
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    @staticmethod
    def _log_likelihood(Y, gamma, sigma):
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        if Ymean == Ym:
            Ym = 0.999 * Ym
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # Look for possible roots
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # All possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # Look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)


class SPOTDetector:
    """
    Wrapper class for SPOT to handle streaming data detection interfaces.
    """

    def __init__(self, q=1e-5, init_data_ratio=1.0, dynamic=True, update_every=1):
        """
        Args:
            q (float): Detection level (risk probability).
            init_data_ratio (float): Ratio of data used for initialization.
            dynamic (bool): Whether to update threshold dynamically.
            update_every (int): Update interval.
        """
        self.q = q
        self.init_data_ratio = init_data_ratio
        self.dynamic = dynamic
        self.update_every = update_every
        self.spot = None
        self.current_index = 0

    def fit(self, init_data):
        """
        Fit the model with initialization data.
        """
        if len(init_data) == 0:
            return

        # Flatten data
        init_data_flat = init_data.flatten()

        # Calculate initialization length
        init_len = int(len(init_data_flat) * self.init_data_ratio)
        if init_len < 10:  # Minimum requirement
            init_len = len(init_data_flat)

        # Initialize SPOT
        self.spot = SPOT(q=self.q)

        # Split into init and stream if needed
        if self.init_data_ratio >= 1.0 or init_len == len(init_data_flat):
            self.spot.fit(init_data_flat, np.array([]))
        else:
            init_train = init_data_flat[:init_len]
            stream_data = init_data_flat[init_len:]
            self.spot.fit(init_train, stream_data)

        # Initialize with level 0.99
        self.spot.initialize(level=0.99, verbose=False)

    def detect_point(self, data_point):
        """
        Detect anomalies for a single data point.
        
        Returns:
            score (float): Anomaly score (distance from threshold).
            threshold (float): Current threshold.
            is_anomaly (int): 0 for normal, 1 for anomaly.
        """
        if self.spot is None:
            return 0.0, np.nan, 0

        # Ensure 1D
        if len(data_point.shape) > 1:
            data_point_value = data_point.flatten()[0]
        else:
            data_point_value = data_point[0]

        current_threshold = self.spot.extreme_quantile

        # Calculate score
        anomaly_score = max(0, data_point_value - current_threshold)

        # Determine anomaly
        is_anomaly = 1 if data_point_value > current_threshold else 0

        # Dynamic update
        if self.dynamic and self.spot.data is not None:
            self.spot.data = np.append(self.spot.data, data_point_value)
            self.spot.n += 1

            if data_point_value > self.spot.init_threshold:
                self.spot.peaks = np.append(self.spot.peaks, data_point_value - self.spot.init_threshold)
                self.spot.Nt += 1

                # Update threshold periodically
                if self.current_index % self.update_every == 0:
                    g, s, l = self.spot._grimshaw()
                    self.spot.extreme_quantile = self.spot._quantile(g, s)

        self.current_index += 1
        return anomaly_score, current_threshold, is_anomaly