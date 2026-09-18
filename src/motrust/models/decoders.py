"""
Decoder modules for RNA and ATAC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, NegativeBinomial, Bernoulli


class BaseDecoder(nn.Module):
    """Base decoder."""
    
    def __init__(self, latent_dim, output_dim, hidden_dims, dropout=0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build the decoder network.
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        h = self.decoder(z)
        return h


class RNADecoder(BaseDecoder):
    """
    RNA decoder supporting poisson, nb and zinb distributions.
    - poisson: Predict lambda.
    - nb: Predict mu and theta.
    - zinb: Predict mu, theta and pi.
    """
    
    def __init__(self, latent_dim, output_dim, hidden_dims=[256, 512], dropout=0.1, distribution: str = "zinb"):
        super().__init__(latent_dim, output_dim, hidden_dims, dropout)
        
        self.distribution = distribution.lower()
        if self.distribution == "poisson":
            self.fc_lambda = nn.Linear(hidden_dims[-1], output_dim)
        elif self.distribution == "nb":
            self.fc_mu = nn.Linear(hidden_dims[-1], output_dim)
            self.fc_theta = nn.Linear(hidden_dims[-1], output_dim)
        elif self.distribution == "zinb":
            self.fc_mu = nn.Linear(hidden_dims[-1], output_dim)
            self.fc_theta = nn.Linear(hidden_dims[-1], output_dim)
            self.fc_pi = nn.Linear(hidden_dims[-1], output_dim)
        else:
            raise ValueError(f"Unsupported RNA distribution: {distribution}")
    
    def forward(self, z):
        h = self.decoder(z)
        if self.distribution == "poisson":
            lambda_param = F.softplus(self.fc_lambda(h)) + 1e-6
            lambda_param = torch.nan_to_num(lambda_param, nan=1e-6, posinf=1e6, neginf=1e-6).clamp(1e-6, 1e6)
            return lambda_param
        
        mu = F.softplus(self.fc_mu(h)) + 1e-4          # ensure >0
        theta = F.softplus(self.fc_theta(h)) + 1e-4     # dispersion >0
        
        # Ensure numerical stability.
        mu = torch.nan_to_num(mu, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
        theta = torch.nan_to_num(theta, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
        
        if self.distribution == "nb":
            return mu, theta
        else:
            pi = torch.sigmoid(self.fc_pi(h))
            pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1 - 1e-6)
            return mu, theta, pi
    
    @staticmethod
    def poisson_log_prob(x, lam, eps: float = 1e-8):
        lam = torch.clamp(lam, min=eps)
        return x * torch.log(lam) - lam - torch.lgamma(x + 1)

    @staticmethod
    def nb_log_prob(x, mu, theta, eps: float = 1e-8):
        mu = torch.clamp(mu, min=eps)
        theta = torch.clamp(theta, min=eps)
        log_theta_mu = torch.log(theta + mu)
        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
        t2 = theta * (torch.log(theta) - log_theta_mu)
        t3 = x * (torch.log(mu) - log_theta_mu)
        return t1 + t2 + t3
    
    @staticmethod
    def zinb_log_prob(x, mu, theta, pi, eps: float = 1e-8):
        mu = torch.clamp(mu, min=eps)
        theta = torch.clamp(theta, min=eps)
        pi = torch.clamp(pi, min=eps, max=1 - eps)
        
        log_theta_mu = torch.log(theta + mu)
        # NB log-prob
        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
        t2 = theta * (torch.log(theta) - log_theta_mu)
        t3 = x * (torch.log(mu) - log_theta_mu)
        log_nb = t1 + t2 + t3
        
        log_nb_zero = theta * (torch.log(theta) - log_theta_mu)  # x=0
        log_zero = torch.logaddexp(torch.log(pi), torch.log1p(-pi) + log_nb_zero)
        log_prob = torch.where(x < eps, log_zero, torch.log1p(-pi) + log_nb)
        return log_prob


class ATACDecoder(BaseDecoder):
    """ATAC decoder with a Bernoulli distribution."""
    
    def __init__(self, latent_dim, output_dim, hidden_dims=[256, 512], dropout=0.1):
        super().__init__(latent_dim, output_dim, hidden_dims, dropout)
        
        # Bernoulli distribution parameters: logits.
        self.fc_logits = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, z):
        h = self.decoder(z)
        logits = self.fc_logits(h)
        # Prevent extreme values from passing NaN/Inf to the Bernoulli distribution.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = torch.clamp(logits, -30.0, 30.0)
        return logits
    
    def get_distribution(self, logits):
        """Return the Bernoulli distribution."""
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = torch.clamp(logits, -30.0, 30.0)
        return Bernoulli(logits=logits)


class ADTDecoder(BaseDecoder):
    """ADT decoder with Poisson (default) or negative binomial (mu, theta) output."""

    def __init__(self, latent_dim, output_dim, hidden_dims=[256, 512], dropout=0.1, distribution: str = "poisson"):
        super().__init__(latent_dim, output_dim, hidden_dims, dropout)
        self.distribution = distribution.lower()
        if self.distribution == "poisson":
            self.fc_lambda = nn.Linear(hidden_dims[-1], output_dim)
        elif self.distribution == "nb":
            self.fc_mu = nn.Linear(hidden_dims[-1], output_dim)
            self.fc_theta = nn.Linear(hidden_dims[-1], output_dim)
        else:
            raise ValueError(f"Unsupported ADT distribution: {distribution}")

    def forward(self, z):
        h = self.decoder(z)
        if self.distribution == "poisson":
            lambda_param = F.softplus(self.fc_lambda(h)) + 1e-6
            lambda_param = torch.nan_to_num(lambda_param, nan=1e-6, posinf=1e6, neginf=1e-6).clamp(1e-6, 1e6)
            return lambda_param

        mu = F.softplus(self.fc_mu(h)) + 1e-4
        theta = F.softplus(self.fc_theta(h)) + 1e-4
        mu = torch.nan_to_num(mu, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
        theta = torch.nan_to_num(theta, nan=1e-4, posinf=1e6, neginf=1e-4).clamp(1e-4, 1e6)
        return mu, theta

    @staticmethod
    def poisson_log_prob(x, lam, eps: float = 1e-8):
        lam = torch.clamp(lam, min=eps)
        return x * torch.log(lam) - lam - torch.lgamma(x + 1)

    @staticmethod
    def nb_log_prob(x, mu, theta, eps: float = 1e-8):
        mu = torch.clamp(mu, min=eps)
        theta = torch.clamp(theta, min=eps)
        log_theta_mu = torch.log(theta + mu)
        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
        t2 = theta * (torch.log(theta) - log_theta_mu)
        t3 = x * (torch.log(mu) - log_theta_mu)
        return t1 + t2 + t3
