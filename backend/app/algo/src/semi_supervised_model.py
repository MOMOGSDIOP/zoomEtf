"""
Modèle semi-supervisé avec VAE pour ETFs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ETFSemiSupervisedModel(nn.Module):
    def __init__(self, input_dim=25, dropout_rate=0.3, combined_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.combined_dim = combined_dim or input_dim
        
        # Encoder pour les features de base
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Encoder pour les features combinées
        if combined_dim and combined_dim != input_dim:
            self.combined_encoder = nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.LeakyReLU(),
                nn.LayerNorm(256),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            )
        
        # Têtes communes
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
            nn.Linear(128, input_dim)  # Reconstruction des features originales
        )

    def forward(self, x, supervised=True, use_combined=False):
        # Choix de l'encodeur
        if use_combined and hasattr(self, 'combined_encoder'):
            encoded = self.combined_encoder(x)
        else:
            encoded = self.base_encoder(x)
        
        if supervised:
            return self.supervised_head(encoded)
        else:
            mu = self.fc_mu(encoded)
            logvar = self.fc_var(encoded)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std