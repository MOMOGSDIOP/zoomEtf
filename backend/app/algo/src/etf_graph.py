from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch_geometric.data import Data
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ETFGraphConfig:
    normalize_features: bool = True
    use_alternative_data: bool = False
    use_temporal_features: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class ETFGraphProcessor:
    """Processeur de graphe ETF aligné avec le dataset complet"""
    
    def __init__(self, config: ETFGraphConfig):
        self.config = config
        self.feature_stats = {}
        
    def build_graph_from_raw(self, etf_data: List[Dict[str, Any]]) -> Data:
        """Construction du graphe avec gestion robuste des erreurs"""
        try:
            # 1. Extraction des éléments de base
            base_features, edge_index, etf_ids, asset_ids = self._extract_base_elements(etf_data)
            
            # 2. Extraction des features avancées
            alt_features = self._extract_alternative_features(etf_data) if self.config.use_alternative_data else None
            temp_features = self._extract_temporal_features(etf_data) if self.config.use_temporal_features else None
            
            # 3. Combinaison des features
            all_features = self._combine_features(base_features, alt_features)
            
            # 4. Normalisation
            if self.config.normalize_features and len(all_features) > 0:
                all_features = self._normalize_features(all_features)
                
            # 5. Construction du graphe
            return Data(
                x=torch.FloatTensor(all_features).to(self.config.device),
                edge_index=torch.LongTensor(edge_index).t().contiguous().to(self.config.device),
                etf_ids=etf_ids,
                asset_ids=asset_ids,
                edge_attr=self._get_edge_weights(edge_index, etf_data),
                temporal_features=temp_features
            )
            
        except Exception as e:
            logger.error(f"Graph construction failed: {str(e)}")
            raise

    def _extract_base_elements(self, etf_data: List[Dict[str, Any]]) -> Tuple:
        """Extraction des éléments de base avec validation des données"""
        node_features = []
        edge_index = [[], []]
        etf_ids = []
        asset_ids = []
        node_mapping = {}
        current_idx = 0
        
        # ETF nodes
        for etf in etf_data:
            try:
                etf_id = etf['etfId']
                node_mapping[f"etf_{etf_id}"] = current_idx
                etf_ids.append(etf_id)
                
                # Features de base avec validation
                fundamentals = etf.get('fundamentals', {})
                risk = etf.get('riskAnalysis', {})
                
                node_features.append([
                    fundamentals.get('costs', {}).get('ter', 0.0),
                    fundamentals.get('liquidity', {}).get('avgDailyVolume', 0),
                    risk.get('volatility', {}).get('annualized', 0.0),
                    fundamentals.get('priceData', {}).get('premiumDiscount', 0.0),
                    risk.get('drawdowns', {}).get('maxDrawdown', 0.0)
                ])
                current_idx += 1
            except KeyError as e:
                logger.warning(f"Missing required ETF data field: {str(e)}")
                continue
        
        # Asset nodes
        for etf in etf_data:
            try:
                holdings = etf.get('portfolio', {}).get('holdings', [])
                for holding in holdings:
                    asset_id = holding.get('assetId')
                    if not asset_id:
                        continue
                        
                    asset_key = f"asset_{asset_id}"
                    if asset_key not in node_mapping:
                        node_mapping[asset_key] = current_idx
                        asset_ids.append(asset_id)
                        
                        # Features des actifs
                        node_features.append([
                            holding.get('weight', 0.0),
                            holding.get('contributionToTrackingError', 0.0),
                            1 if holding.get('sector') == 'Technology' else 0,
                            1 if holding.get('country', 'US') == 'US' else 0
                        ])
                        current_idx += 1
                    
                    # Connection ETF -> Actif
                    edge_index[0].append(node_mapping[f"etf_{etf['etfId']}"])
                    edge_index[1].append(node_mapping[asset_key])
            except Exception as e:
                logger.warning(f"Error processing holdings for ETF {etf.get('etfId', 'unknown')}: {str(e)}")
                continue
        
        return np.array(node_features, dtype=np.float32), edge_index, etf_ids, asset_ids

    def _extract_alternative_features(self, etf_data: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extraction des données alternatives avec validation"""
        features = []
        for etf in etf_data:
            try:
                alt_data = etf.get('alternativeData', {})
                peer_data = etf.get('peerComparison', {})
                
                features.append([
                    np.clip(alt_data.get('sentiment', {}).get('newsSentiment', 0.5), 0, 1),
                    alt_data.get('flows', {}).get('30dNetFlow', 0) / 1e9,
                    np.clip(peer_data.get('percentileRank', {}).get('cost', 0.5), 0, 1)
                ])
            except Exception as e:
                logger.warning(f"Alternative data error for ETF {etf.get('etfId', 'unknown')}: {str(e)}")
                features.append([0.5, 0.0, 0.5])
        
        return np.array(features, dtype=np.float32) if features else None

    def _extract_temporal_features(self, etf_data: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Extraction des features temporelles"""
        temporal_features = []
        for etf in etf_data:
            try:
                # Statistiques des returns
                returns = np.array(etf.get('timeSeries', {}).get('dailyReturns', [0]))
                feat = [
                    float(np.mean(returns)),
                    float(np.std(returns)),
                    float(np.min(returns)),
                    float(np.max(returns)),
                    float(np.percentile(returns, 25)),
                    float(np.percentile(returns, 75))
                ]
                
                # Valeur NAV
                nav_history = etf.get('timeSeries', {}).get('historicalNav', [])
                if nav_history:
                    last_nav = nav_history[-1].get('value', nav_history[-1]) if isinstance(nav_history[-1], dict) else nav_history[-1]
                    feat.append(float(last_nav))
                else:
                    feat.append(float(etf.get('fundamentals', {}).get('priceData', {}).get('currentPrice', 0)))
                
                temporal_features.append(feat)
            except Exception as e:
                logger.warning(f"Temporal data error for ETF {etf.get('etfId', 'unknown')}: {str(e)}")
                temporal_features.append([0.0]*7)
        
        return torch.FloatTensor(temporal_features).to(self.config.device) if temporal_features else None

    def _combine_features(self, *feature_sets: List[np.ndarray]) -> np.ndarray:
        """Combinaison robuste des features"""
        valid_sets = [fs for fs in feature_sets if fs is not None and len(fs) > 0]
        
        if not valid_sets:
            return np.array([], dtype=np.float32)
        
        # Validation des dimensions
        num_samples = len(valid_sets[0])
        if not all(len(fs) == num_samples for fs in valid_sets):
            raise ValueError("Inconsistent feature dimensions")
            
        return np.concatenate(valid_sets, axis=1).astype(np.float32)

    def _get_edge_weights(self, edge_index: List[List[int]], etf_data: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Calcul optimisé des poids des edges"""
        if not edge_index or len(edge_index[0]) == 0:
            return None
            
        try:
            num_etfs = len(etf_data)
            weights = np.zeros(len(edge_index[0]), dtype=np.float32)
            
            # Pré-calcul des mappings
            etf_mapping = {etf['etfId']: i for i, etf in enumerate(etf_data)}
            asset_mapping = {}
            global_idx = num_etfs
            
            for etf in etf_data:
                for holding in etf.get('portfolio', {}).get('holdings', []):
                    asset_id = holding.get('assetId')
                    if asset_id and asset_id not in asset_mapping:
                        asset_mapping[asset_id] = global_idx
                        global_idx += 1
            
            # Remplissage des poids
            for i, (src, dst) in enumerate(zip(*edge_index)):
                if src >= num_etfs:
                    continue
                    
                asset_id = next((k for k, v in asset_mapping.items() if v == dst), None)
                if asset_id:
                    holdings = etf_data[src].get('portfolio', {}).get('holdings', [])
                    weight = next((h.get('weight', 0.0) for h in holdings if h.get('assetId') == asset_id))
                    weights[i] = weight
            
            # Normalisation
            if np.any(weights):
                min_w, max_w = np.min(weights), np.max(weights)
                if max_w > min_w:
                    weights = (weights - min_w) / (max_w - min_w)
            
            return torch.FloatTensor(weights).to(self.config.device)
            
        except Exception as e:
            logger.error(f"Edge weights error: {str(e)}")
            return None

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalisation MinMax des features"""
        if len(features) == 0:
            return features
            
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0  # Éviter la division par zéro
        
        return (features - min_vals) / ranges