import numpy as np
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from scipy import linalg
from sklearn.utils.extmath import row_norms

class Gaussian_mixture():
    def __init__(self, n_components, means_init = None, weights_init = None, precisions_init = None, random_state = 0, cov_type = 'isotropic', tol = 1e-4, max_iter = 100):
        assert cov_type in ['diag', 'isotropic']
        self.K = n_components
        self.random_state = random_state
        self.means_init = means_init
        self.weights_init = weights_init
        self.precisions_init = precisions_init
        self.cov_type = cov_type
        self.reg_covar = 1e-6
        self.tol = tol
        self.max_iter = max_iter

    def _init_params(self, X):
        resp = np.zeros((X.shape[0], self.K))
        label = (
            KMeans(
                n_clusters=self.K, n_init = 1, random_state=self.random_state
                ).fit(X).labels_)
        
        resp[np.arange(X.shape[0]), label] = 1

        self._initialize(X, resp)
    
    def _initialize(self, X, resp):
        n_samples, _ = X.shape
        weights, means, covariances = None, None, None
        if resp is not None:
            weights, means, covariances = self._estimate_gaussian_parameters(
                X, resp, self.reg_covar
            )
            if self.weights_init is None:
                weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init
        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = self._compute_precision_cholesky(
                covariances
            )
        else:
            self.precisions_cholesky_ = self._compute_precision_cholesky_from_precisions(
                self.precisions_init
            )

        # print('asdfasddfa')
        # print(self.precisions_cholesky_)

    def _flipudlr(self, array):
        return np.flipud(np.fliplr(array))

    def _compute_precision_cholesky_from_precisions(self, precisions):
        precisions_cholesky = np.sqrt(precisions)
        return precisions_cholesky

    def _estimate_gaussian_parameters(self, X, resp, reg_covar):
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        covariances = {
            "diag": self._estimate_gaussian_covariances_diag,
            "isotropic": self._estimate_gaussian_covariances_isotropic,
        }[self.cov_type](resp, X, nk, means, reg_covar)
        return nk, means, covariances
    
    def _estimate_gaussian_covariances_diag(self, resp, X, nk, means, reg_covar):
        avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
        avg_means2 = means**2
        return avg_X2 - avg_means2 + reg_covar

    def _estimate_gaussian_covariances_isotropic(self, resp, X, nk, means, reg_covar):
            N, D = X.shape
            K = resp.shape[1]

            total_sq = 0.0
            for k in range(K):
                diff = X - means[k]
                sq_dist = np.sum(diff**2, axis=1)
                total_sq += np.dot(resp[:, k], sq_dist)
            denom = D * np.sum(nk)

            var = total_sq / denom
            var += reg_covar
            return var

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
        )

    def _estimate_log_prob_resp(self, X):
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _compute_log_det_cholesky(self, matrix_chol, n_features):
        if self.cov_type == "diag":
            log_det_chol = np.sum(np.log(matrix_chol), axis=1)

        elif self.cov_type == "isotropic":
            prec_chol = matrix_chol.item()
            log_det_ = n_features * np.log(prec_chol)
            log_det_chol = np.full(self.K, log_det_)
        return log_det_chol

    def _estimate_log_gaussian_prob(self, X, means, precisions_chol):
        n_samples, n_features = X.shape
        n_components, _ = means.shape
        log_det = self._compute_log_det_cholesky(precisions_chol, n_features)

        if self.cov_type == "diag":
            precisions = precisions_chol**2
            log_prob = (
                np.sum((means**2 * precisions), 1)
                - 2.0 * np.dot(X, (means * precisions).T)
                + np.dot(X**2, precisions.T)
            )
        elif self.cov_type == "isotropic":
            precision = precisions_chol**2
            log_prob = np.empty((n_samples, n_components), dtype=X.dtype)
            for k, mu in enumerate(means):
                diff = X - mu
                log_prob[:, k] = row_norms(diff, squared=True) * precision

        return -0.5 * (n_features * np.log(2 * np.pi).astype(X.dtype) + log_prob) + log_det

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _estimate_log_prob(self, X):
        return self._estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_
        )
    
    def _estimate_weighted_log_prob(self, X):
        return self._estimate_log_prob(X) + self._estimate_log_weights()
    
    def _e_step(self, X):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp
    
    def _m_step(self, X, log_resp):
        self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = self._compute_precision_cholesky(
            self.covariances_
        )

    def _compute_precision_cholesky(self, covariances):
        dtype = covariances.dtype

        if self.cov_type == 'diag':
            precisions_chol = 1.0 / np.sqrt(covariances)
        elif self.cov_type == "isotropic":
            if np.isscalar(covariances):
                var_value = covariances
            else:
                var_value = covariances.item()
            prec_chol_value = 1.0 / np.sqrt(var_value)
            precisions_chol = np.array([prec_chol_value], dtype=dtype)
        return precisions_chol

    def fit(self, X):
        max_lower_bound = -np.inf
        self.converged_ = False

        self._init_params(X)

        lower_bound = -np.inf 

        for n_iter in range(1, self.max_iter + 1):
            prev_lower_bound = lower_bound

            log_prob_norm, log_resp = self._e_step(X)
            self._m_step(X, log_resp)
            lower_bound = log_prob_norm

            change = lower_bound - prev_lower_bound

            if abs(change) < self.tol:
                self.converged_ = True
                break

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_n_iter = n_iter
                self.converged_ = True

        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound
        _, log_resp = self._e_step(X)
        self.precisions_ = 1/self.covariances_

        return log_resp.argmax(axis=1)
    
    def predict(self, X):
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def score_samples(self, X):
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)
    
    def score(self, X):
        return self.score_samples(X).mean()