"""
Modèle semi-supervisé avec VAE pour ETFs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ETFSemiSupervisedModel(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.supervised_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.fc_mu = nn.Linear(128, 64)
        self.fc_var = nn.Linear(128, 64)
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, input_dim)
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, supervised=True):
        encoded = self.encoder(x)
        
        if supervised:
            return self.supervised_head(encoded)
        else:
            mu = self.fc_mu(encoded)
            logvar = self.fc_var(encoded)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar
