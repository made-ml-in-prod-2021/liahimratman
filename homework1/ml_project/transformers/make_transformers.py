import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.validation import (check_is_fitted,
                                FLOAT_DTYPES)


class StandardScalerTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, mean=None, scale=None):
        self._reset()
        return self.partial_fit(X, mean, scale)

    def partial_fit(self, X, mean, scale):
        if mean:
            if X.shape[1] != len(mean):
                print('X', X.shape[1], len(mean))
                raise NotImplementedError()
            self.mean_ = mean
            self.scale_ = scale
        else:
            first_call = not hasattr(self, "n_samples_seen_")
            X = self._validate_data(X, estimator=self, dtype=FLOAT_DTYPES,
                                    force_all_finite='allow-nan', reset=first_call)
            n_features = X.shape[1]
            dtype = np.int64

            if not hasattr(self, 'n_samples_seen_'):
                self.n_samples_seen_ = np.zeros(n_features, dtype=dtype)
            elif np.size(self.n_samples_seen_) == 1:
                self.n_samples_seen_ = np.repeat(
                    self.n_samples_seen_, X.shape[1])
                self.n_samples_seen_ = \
                    self.n_samples_seen_.astype(dtype, copy=False)

            if not hasattr(self, 'scale_'):
                self.mean_ = .0
                self.var_ = .0
            self.mean_, self.var_, self.n_samples_seen_ = \
                _incremental_mean_and_var(X, self.mean_, self.var_,
                                          self.n_samples_seen_)

            if np.ptp(self.n_samples_seen_) == 0:
                self.n_samples_seen_ = self.n_samples_seen_[0]

            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))

        return self

    def transform(self, X, copy=None):
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, copy=copy, reset=False,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')
        X -= self.mean_
        X /= self.scale_

        return X
