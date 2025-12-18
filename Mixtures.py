import torch
import torch.nn.functional as F

class Multinomial_Mixture():
    def __init__(self, data, K, max_iter=100, tole = 1e-6, random_state = 0, is_random = True):
        self.data = data
        self.N, self.cate_dim = data.shape
        self.K = K
        self.max_iter = max_iter
        self.tole = tole
        self.random_state = random_state
        self.is_random = is_random
        self.num_categories = torch.max(data, dim=0).values + 1
        self.one_hot_list = [F.one_hot(data[:, d], num_classes=self.num_categories[d]).double() for d in range(self.cate_dim)]

    def init_em(self):
        if self.is_random:
            g = torch.Generator()
            g.manual_seed(self.random_state)
            self.pi = torch.softmax(torch.randn(self.K, dtype=torch.float64, generator = g), dim=0)
            self.theta = [torch.softmax(torch.randn(self.K, c, dtype=torch.float64, generator = g), dim=1) for c in self.num_categories]
        else:
            self.pi = torch.softmax(torch.ones(self.K, dtype=torch.float64), dim=0)
            self.theta = [torch.softmax(torch.ones(self.K, c, dtype=torch.float64), dim=1) for c in self.num_categories]

    def e_step(self):
        log_pi = torch.log(self.pi).unsqueeze(0)
        for d in range(self.cate_dim):
            X_d = self.one_hot_list[d]
            theta_d = self.theta[d]
            log_theta_d = torch.log(theta_d + 1e-12)
            if d == 0:
                log_prob = (X_d.unsqueeze(1) * log_theta_d.unsqueeze(0)).sum(dim = -1)
            else:
                log_prob += (X_d.unsqueeze(1) * log_theta_d.unsqueeze(0)).sum(dim = -1)
        log_prob = log_pi + log_prob
        log_prob = log_prob - log_prob.logsumexp(dim=1, keepdim=True)
        gamma = torch.exp(log_prob)
        return gamma

    def m_step(self, gamma):
        N_k = gamma.sum(dim=0)
        self.pi = N_k / self.N

        for d in range(self.cate_dim):
            X_d = self.one_hot_list[d]
            weighted = gamma.unsqueeze(2) * X_d.unsqueeze(1)
            theta_d = weighted.sum(dim=0) / N_k.unsqueeze(1)
            self.theta[d] = theta_d

    def log_likelihood(self):
        log_pi = torch.log(self.pi).unsqueeze(0)

        for d in range(self.cate_dim):
            X_d = self.one_hot_list[d]
            log_theta_d = torch.log(self.theta[d] + 1e-12)
            if d == 0:
                log_prob = (X_d.unsqueeze(1) * log_theta_d.unsqueeze(0)).sum(axis = -1)
            else:
                log_prob += (X_d.unsqueeze(1) * log_theta_d.unsqueeze(0)).sum(axis = -1)
        log_prob = log_pi + log_prob
        return torch.logsumexp(log_prob, dim=1).mean()

    def fit(self):
        self.init_em()
        ll_prev = -torch.inf
        self.ll_list = []

        converged = False
        ith = 0

        while not converged:
            gamma = self.e_step()
            self.m_step(gamma)

            ll = self.log_likelihood()
            self.ll_list.append(ll.item())

            if ith == self.max_iter:
                converged = True
            if (ll - ll_prev).abs() < self.tole:
                converged = True
            
            ith += 1
            ll_prev = ll