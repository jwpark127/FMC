import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.utils import logits_to_probs, probs_to_logits
from utils.utils_mode import sorted_counter, to_numpy, calculate_balance, calculate_Fscore, get_min_index, re_calc_distance
import numpy as np
from collections import deque
from EM_numpy import Gaussian_mixture as em_numpy
from Mixtures import Multinomial_Mixture as MM

torch.set_default_dtype(torch.float64)

class Gaussian_mixture():
    def __init__(self, n_components, 
                 sensitive, data_cont = None, data_cate = None, 
                 means_init = None, weights_init = None, precisions_init = None, random_state = 0, 
                 cov_type = 'isotropic', tol = 1e-4, max_iter = 100, device = 'cpu',
                 lr = 1e-2,
                 m_tole = 1e-6,
                 lmda = 0,
                 sub_lmda = 1e-2,
                 m_step_iter = 1000,
                 window_length = 100,
                 verbose = False,
                 train_cov = True,
                 start_em = True,
                 em_means_init = None, em_weigths_init = None, em_precisions_init = None
                 ):
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
        self.device = device
        self.lr = lr
        self.m_tole = m_tole
        self.lmda = lmda
        self.sensitive = sensitive
        self.S = self.sensitive.unique()
        self.n_sensitive = self.S.unique().shape[0]
        self.count_S = torch.tensor(sorted_counter(self.sensitive, self.S.shape[0])[:,1], device = device, dtype = torch.float64)
        self.ratio_S =  self.count_S / self.sensitive.shape[0]
        self.n_samples = self.sensitive.shape[0]
        self.m_step_iter = m_step_iter

        self.conts = data_cont is not None and bool(data_cont.any())
        if self.conts:
            self.data_cont = data_cont.type(torch.float64).to(self.device)
        self.cates = data_cate is not None and bool(data_cate.any())
        if self.cates:
            self.data_cate = data_cate.type(torch.int64).to(self.device)
            self.num_categories = torch.max(data_cate, dim=0).values + 1
            self.data_cate_oh = [F.one_hot(self.data_cate[:, d], num_classes=self.num_categories[d]).double() for d in range(data_cate.shape[1])]

        self.window_length = window_length
        self.verbose = verbose
        self.train_cov = train_cov
        self.start_em = start_em
        self.em_means_init = em_means_init
        self.em_weigths_init = em_weigths_init
        self.em_precisions_init = em_precisions_init
        self.sub_lmda = sub_lmda
        self.sensitive_index = [to_numpy(torch.where(self.sensitive == sth)[0]) for sth in range(self.n_sensitive)]

    def _init_params(self):
        self._initialize()
    
    def _initialize(self):
        self.NLL_list = []
        self.Fscore_list = []
        self.balance_list = []
        self.dist_list = []
        self.predict_list = []

        if self.conts:
            EM = em_numpy(n_components = self.K, cov_type = self.cov_type, 
                        means_init = self.em_means_init,
                        weights_init = self.em_weigths_init,
                        precisions_init = self.em_precisions_init,
                        tol = 1e-5, max_iter = self.max_iter, random_state=self.random_state)

            EM.fit(to_numpy(self.data_cont))
            self.means_ = torch.tensor(EM.means_, device = self.device)
            self.precisions_ = torch.tensor(EM.precisions_, device = self.device)
            self.covariances_ = torch.tensor(EM.covariances_, device = self.device)
            self.precisions_cholesky_ = self._compute_precision_cholesky(self.covariances_)
            cont_weights = torch.tensor(EM.weights_, device = self.device)

        if self.cates:
            mm = MM(self.data_cate.to('cpu', torch.int64), K = self.K, is_random = True, random_state = self.random_state,
                    max_iter=5, tole = 1e-2)
            mm.fit()
            self.cate_logits = [torch.ones([self.K, self.num_categories[d]], device = self.device) for d in range(self.data_cate.shape[1])]
            self.cate_weights = (mm.pi.to(self.device))
        weights_list = []

        if self.conts:
            weights_list.append(cont_weights)
        if self.cates:
            weights_list.append(self.cate_weights)

        if len(weights_list) > 0:
            self.weights_ = torch.vstack(weights_list).mean(axis=0)
        else:
            self.weights_ = None
        self.logits = probs_to_logits(self.weights_)

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
        )

    def _compute_precision_cholesky(self, covariances):
        estimate_precision_error_message = (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "increase reg_covar, or scale the input data."
        )
        dtype = covariances.dtype
        if dtype == torch.float32:
            estimate_precision_error_message += (
                " The numerical accuracy can also be improved by passing float64"
                " data instead of float32."
            )
        if self.cov_type == "diag":
            if torch.any(covariances <= 0.0):
                raise ValueError(estimate_precision_error_message)
            precisions_chol = 1.0 / torch.sqrt(covariances)
        elif self.cov_type == "isotropic":
            precisions_chol = 1.0 / torch.sqrt(covariances)
        return precisions_chol
      
    def _compute_log_det_cholesky(self, matrix_chol, n_features):
        if self.cov_type == "diag":
            log_det_chol = torch.sum(torch.log(matrix_chol), dim=1)
        elif self.cov_type == "isotropic":
            prec_chol = matrix_chol.squeeze()
            log_det_ = n_features * torch.log(prec_chol)
            log_det_chol = torch.ones(self.K, dtype=matrix_chol.dtype, device = self.device) * log_det_
        return log_det_chol
    
    def _estimate_log_gaussian_prob(self, means, precisions_chol):
        n_samples, n_features = self.data_cont.shape
        n_components, _ = means.shape
        log_det = self._compute_log_det_cholesky(precisions_chol, n_features)
        if self.cov_type == "diag":
            precisions = precisions_chol**2
            log_prob = (
                torch.sum((means**2 * precisions), dim=1)
                - 2.0 * torch.mm(self.data_cont, (means * precisions).T)
                + torch.mm(self.data_cont**2, precisions.T)
            )
        elif self.cov_type == "isotropic":
            precision = precisions_chol**2
            log_prob = torch.empty((n_samples, n_components), device = self.device)
            for k, mu in enumerate(means):
                diff = self.data_cont - mu
                log_prob[:, k] = torch.norm(diff, dim=1, p=2) ** 2 * precision
        return -0.5 * (n_features * torch.log(torch.tensor(2 * torch.pi, device = self.device)) + log_prob) + log_det

    def _estimate_log_prob(self):
        return self._estimate_log_gaussian_prob(self.means_, self.precisions_cholesky_)

    def _estimate_log_categorical(self):
        log_categorical_prob = torch.stack([(self.data_cate_oh[d].unsqueeze(1) * torch.log(torch.softmax(self.cate_logits[d], dim = 1).unsqueeze(0))).sum(axis = -1) for d in range(self.data_cate.shape[1])])
        return log_categorical_prob.sum(axis = 0)

    def _estimate_log_weights(self):
        return torch.log(logits_to_probs(self.logits))
        
    def _estimate_weighted_log_prob(self):
        log_weights = self._estimate_log_weights()
        if log_weights.dim() == 1:
            n_samples = self.data_cont.shape[0] if self.conts else self.data_cate.shape[0]
            log_weights = log_weights.unsqueeze(0).expand(n_samples, -1)

        log_prob = log_weights.clone()

        if self.conts:
            log_prob = log_prob + self._estimate_log_prob()
        if self.cates:
            log_prob = log_prob + self._estimate_log_categorical()

        return log_prob
    
    def _estimate_log_prob_resp(self):
        weighted_log_prob = self._estimate_weighted_log_prob()
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        log_resp = weighted_log_prob - log_prob_norm.unsqueeze(1)
        return log_prob_norm, log_resp
        
    def _e_step(self):
        log_prob_norm, log_resp = self._estimate_log_prob_resp()
        return torch.mean(log_prob_norm), log_resp

    def _m_step(self):
        self.logits = torch.nn.Parameter(self.logits, requires_grad=True)
        param_list = [self.logits]
        if self.conts:
            self.means_ = torch.nn.Parameter(self.means_, requires_grad=True)
            param_list.append(self.means_)
            self.root_precisions_ = torch.nn.Parameter(
                torch.sqrt(self.precisions_), requires_grad=True
            )
            if self.train_cov:
                param_list.append(self.root_precisions_)
        if self.cates:
            self.cate_logits = [
                torch.nn.Parameter(logit, requires_grad=True)
                for logit in self.cate_logits
            ]
            param_list.extend(self.cate_logits)
        optimizer = optim.Adam(param_list, lr=self.lr)
        converged = False
        ith = 0
        loss_before = torch.inf
        _, log_resp = self._e_step()
        resp_detach = torch.exp(log_resp).detach()      
        differ_window = deque(list(range(self.window_length)))
        
        while not converged:
            ith += 1
            if self.conts:
                covariances_ = (1/self.root_precisions_**2)
                self.precisions_cholesky_ = self._compute_precision_cholesky(covariances_)
            optimizer.zero_grad()
            clustering_loss = - (resp_detach * self._estimate_weighted_log_prob()).mean()
            Log_like, resp_ = self._e_step()
            resp_ = torch.exp(resp_)
            fair_loss = resp_detach.mean(axis = 0) * torch.abs(torch.stack([(resp_[sens_idx].sum(axis = 0) / resp_.sum(axis = 0) - self.ratio_S[sth]) 
                                               for sth, sens_idx in enumerate(self.sensitive_index)]))
            
            cluster_size_penalty = torch.stack([self.ratio_S[sth] * 1/(resp_[sens_idx].sum(axis = 0) + 1e-6) for sth, sens_idx in enumerate(self.sensitive_index)])
            clusters_ = np.array(self.assignment())
            target_cluster = get_min_index((clusters_))
            total_loss = clustering_loss + self.lmda * fair_loss.max() + self.sub_lmda * cluster_size_penalty[target_cluster]
            self.logits_before = self.logits
            total_loss.backward()
            optimizer.step()
            self.loss_differ = loss_before - total_loss
            differ_window.popleft()
            differ_window.append(torch.abs(self.loss_differ))
            mean_differ = torch.tensor(differ_window).mean()
            if torch.abs(mean_differ) < self.m_tole:
                converged = True
                self.converged = converged

            if ith >= self.m_step_iter:
                converged = True
                self.converged = converged
            loss_before = total_loss
            print(f'[{self.n_iter}-{ith}] Log-likelihood: {Log_like:0.4f}, Delta: {calculate_Fscore(resp_.detach().cpu(), self.sensitive): 0.4f}, Balance: {calculate_balance(self, self.sensitive): 0.4f}', end = '\r')
        self.logits = self.logits.detach()
        self.weights_ = logits_to_probs(self.logits)

        if self.conts:
            self.means_ = self.means_.detach()
            self.root_precisions_ = self.root_precisions_.detach()
            self.precisions_ = self.root_precisions_**2
            self.covariances_ = (1/self.root_precisions_**2)
            self.precisions_cholesky_ = self._compute_precision_cholesky(self.covariances_)

    def fit(self):
        max_lower_bound = -torch.inf
        self.converged_ = False

        self._init_params()

        lower_bound = -torch.inf 
        n_iter = 1
        while not self.converged_:
            prev_lower_bound = lower_bound
            
            self.converged = False
            self.over_trained = False
            self.n_iter = n_iter
            self._m_step()
            
            lower_bound, log_resp = self._e_step()
            resp = torch.exp(log_resp)

            change = lower_bound - prev_lower_bound
    
            self.NLL_list.append(-lower_bound.item())
            self.balance_list.append(calculate_balance(self, self.sensitive))
            self.Fscore_list.append(calculate_Fscore(resp.detach().cpu(), self.sensitive))
            if self.conts:
                self.dist_list.append(re_calc_distance(self.data_cont, self.predict(), self.K)[1])
            self.predict_list.append(to_numpy(self.predict()).tolist())

            n_iter += 1

            if lower_bound > max_lower_bound or max_lower_bound == -torch.inf:
                max_lower_bound = lower_bound
                best_n_iter = n_iter

            if abs(change) < self.tol:
                self.converged_ = True
            
            if n_iter >= self.max_iter:
                self.converged_ = True
        self.n_iter_ = best_n_iter
        self.lower_bound_ = lower_bound.item()

        _, log_resp = self._e_step()

        self.NLL_list = np.array(self.NLL_list)
        self.balance_list = np.array(self.balance_list)
        self.Fscore_list = np.array(self.Fscore_list)
        self.predict_list = (self.predict_list)
        best_balance_index = np.where(self.balance_list == np.nanmax(self.balance_list))[0][0]
        
        if self.conts:
            self.precisions_ = 1/self.covariances_
            self.dist_list = np.array(self.dist_list)
            self.dist = self.dist_list[best_balance_index]
        self.NLL = self.NLL_list[best_balance_index]
        self.balance = self.balance_list[best_balance_index]
        self.Fscore = self.Fscore_list[best_balance_index]
        self.predict_ = self.predict_list[best_balance_index]
        
        if self.conts:
            self.params = self._get_parameters()
        print('\n')
        return log_resp.argmax(dim=1)
    
    def predict(self):
        return self._estimate_weighted_log_prob().argmax(dim=1)
    
    def assignment(self):
        result = self.predict()
        clusters = [sorted_counter(result[sens_idx], self.K)[:,1] for sens_idx in self.sensitive_index]
        return clusters