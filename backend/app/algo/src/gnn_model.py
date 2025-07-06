import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ETFGraphModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, heads: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim

        # D1: Couche GNN (GAT - Graph Attention Network)
        self.gnn = GATConv(input_dim, hidden_dim, heads=heads, concat=False)

        # D2: Module Temporel LSTM
        self.temporal = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # D3: Corrélation pairwise des embeddings
        self.correlation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # D4: Estimation d'incertitude (mu, sigma)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # mean and std
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_data):
        x, edge_index = graph_data.x, graph_data.edge_index

        # D1: GNN - Encode les relations entre ETFs dans le graphe
        x = self.gnn(x, edge_index)
        x = self.dropout(x)

        # D2: Traitement temporel - Données séquentielles (via LSTM)
        x_lstm, _ = self.temporal(x.unsqueeze(0))  # batch_size = 1
        x = x_lstm.squeeze(0)  # Retire la dimension batch

        # D3: Corrélations entre embeddings
        corr_matrix = self.calculate_correlations(x)  # [N, N]

        # D4: Estimation d'incertitude (mu, sigma pour chaque ETF)
        mu_sigma = self.uncertainty_head(x)  # [N, 2]
        mu, sigma = torch.chunk(mu_sigma, 2, dim=1)  # [N, 1] x2

        return x, corr_matrix, mu, sigma


    def train(self,graph_data: dict,targets: torch.Tensor,downstream_model: nn.Module,loss_fn: nn.Module,optimizer: torch.optim.Optimizer,device: torch.device,memory_optimizer=None ) -> dict:
        """
        Entraîne le GNN + modèle aval (semi-supervisé), et retourne les métriques
        """

        try : 
            # Validation minimal
            if not isinstance(graph_data, dict) or 'x' not in graph_data or 'edge_index' not in graph_data:
                raise ValueError("graph_data doit contenir au minimum les clés 'x' et 'edge_index'")
            
            if not isinstance(targets, torch.Tensor):
                raise ValueError("targets doit être un tensor")
            
            # Passage sur device
            graph_data = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in graph_data.items()}
                
            targets = targets.to(device)
            
            # 1. Passage GNN
            embeddings, corr, mu, sigma = self.forward(graph_data)
            
            if embeddings.shape[0] != graph_data['x'].shape[0]:
                raise RuntimeError(f"Incohérence de dimensions entre embeddings {embeddings.shape} et x {graph_data['x'].shape}")
            
            # 2. Fusion
            combined = torch.cat([embeddings, graph_data['x']], dim=1)
            
            # 3. Entraînement du modèle semi-supervisé avec loss globale
            optimizer.zero_grad()
            preds = downstream_model(combined)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            
            return {"gnn_loss": float(loss.item()),"correlation": float(corr.mean().item())}
        

        finally:
            if memory_optimizer:
                memory_optimizer.clear_tensors(embeddings, corr, mu, sigma, preds)
        

    def calculate_correlations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule une matrice de similarité cosinus entre les embeddings des ETFs.
        Args:
            x (Tensor): [N, D] où N = nombre d'ETFs, D = dimension des embeddings
        Returns:
            Tensor: [N, N] matrice symétrique de similarité (cosine similarity)
        """
        # Normalise chaque vecteur pour que le produit scalaire donne le cosinus
        x_norm = F.normalize(x, p=2, dim=1)  # [N, D]

        # Produit scalaire => similarité cosinus
        similarity_matrix = torch.matmul(x_norm, x_norm.T)  # [N, N], valeurs ∈ [-1, 1]

        return similarity_matrix
