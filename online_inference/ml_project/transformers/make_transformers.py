import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.validation import (check_is_fitted,
                                      FLOAT_DTYPES)


class StandardScalerTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        """
        Init
        :param copy: copy flag
        :param with_mean: use centering flag
        :param with_std: use scaling flag
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.scale_ = None
        self.n_samples_seen_ = None
        self.mean_ = None
        self.var_ = None

    def _reset(self):
        """
        Reset parameters
        :return: None
        """
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, data, mean=None, scale=None):
        """
        Fir transformer
        :param data: data
        :param mean: sample means
        :param scale: sample scales
        :return: fitted transformer
        """
        self._reset()
        return self.partial_fit(data, mean, scale)

    def partial_fit(self, data, mean, scale):
        """
        Fit transformer
        :param data: data
        :param mean: sample means
        :param scale: sample scales
        :return: fitted transformer
        """
        if mean:
            self.mean_ = mean
            self.scale_ = scale
        else:
            first_call = not hasattr(self, "n_samples_seen_")
            data = self._validate_data(data, estimator=self, dtype=FLOAT_DTYPES,
                                       force_all_finite='allow-nan', reset=first_call)
            n_features = data.shape[1]
            dtype = np.int64

            if not hasattr(self, 'n_samples_seen_'):
                self.n_samples_seen_ = np.zeros(n_features, dtype=dtype)
            elif np.size(self.n_samples_seen_) == 1:
                self.n_samples_seen_ = np.repeat(
                    self.n_samples_seen_, data.shape[1])
                self.n_samples_seen_ = \
                    self.n_samples_seen_.astype(dtype, copy=False)

            if not hasattr(self, 'scale_'):
                self.mean_ = .0
                self.var_ = .0
            self.mean_, self.var_, self.n_samples_seen_ = \
                _incremental_mean_and_var(data, self.mean_, self.var_,
                                          self.n_samples_seen_)

            if np.ptp(self.n_samples_seen_) == 0:
                self.n_samples_seen_ = self.n_samples_seen_[0]

            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))

        return self

    def transform(self, data, copy=False):
        """
        Transform data
        :param data: data
        :param copy: copy flag
        :return: transformed data
        """
        check_is_fitted(self)

        copy = copy if not copy else self.copy
        data = self._validate_data(data, copy=copy, reset=False,
                                   estimator=self, dtype=FLOAT_DTYPES,
                                   force_all_finite='allow-nan')
        data -= self.mean_
        data /= self.scale_

        return data
