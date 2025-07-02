"""
Fonctions de loss avancées pour la notation d'ETFs
"""
import torch
import torch.nn as nn

class ETFCompositeLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3):
        super().__init__()
        self.alpha = alpha  # Poids pour la loss supervisée
        self.beta = beta    # Poids pour la KL divergence
        self.mse = nn.MSELoss()
        
    def forward(self, preds, targets, vae_outputs=None):
        # Loss supervisée
        supervised_loss = self.mse(preds, targets)
        
        if vae_outputs is None:
            return supervised_loss
            
        # Loss non supervisée (VAE)
        recon_x, mu, logvar = vae_outputs
        recon_loss = F.mse_loss(recon_x, original_input, reduction='mean')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return (self.alpha * supervised_loss + 
                (1-self.alpha) * recon_loss + 
                self.beta * kld_loss)