import pandas as pd
import numpy as np
import torch
from sklearn.feature_selection import SelectKBest, f_regression
import  logging
from config import REQUIRED_COLUMNS, COLUMN_MAPPING
logger = logging.getLogger(__name__)

class ETFScoring:
    RATING_SCALE = {
        'D': (0.0, 0.4),
        'C': (0.4, 0.6),
        'B': (0.6, 0.8),
        'A': (0.8, 0.9),
        'A+': (0.9, 1.0)  # Note: A+ inclut 1.0
    }
    
    def __init__(self, model, device, monitor, gnn_model,feature_selector,input_dim=25):
        self.model = model
        self.device = device
        self.monitor = monitor
        self.input_dim = input_dim
        self.gnn_model = gnn_model
        self.feature_selector = feature_selector
    

    
    def _prepare_etf_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sélectionne et réduit les features à la dimension attendue"""
        num_cols = X.select_dtypes(include=np.number).columns
        
        # Création du sélecteur de features si nécessaire
        if self.feature_selector is None:
            # Utilisation de valeurs fictives pour l'initialisation
            dummy_scores = np.random.rand(len(X))
            self.feature_selector = SelectKBest(f_regression, k=self.input_dim)
            self.feature_selector.fit(X[num_cols], dummy_scores)
        
        # Sélection des features importantes
        selected_features = self.feature_selector.transform(X[num_cols])
        return pd.DataFrame(selected_features, columns=[f"feature_{i}" for i in range(self.input_dim)])
    

    def predict(self, X:pd.DataFrame) -> pd.DataFrame:
        """Prédiction avec échelle de notes fixe"""
        self.monitor.log_operation_start('prediction')
        
        try:
            # Rename columns to match expected names
            X_renamed = X.rename(columns=COLUMN_MAPPING)
            
            # 1. Vérification des colonnes REQUISES
            required_cols = REQUIRED_COLUMNS 
            missing = set(required_cols) - set(X_renamed.columns)
            
            # 2. Vérification des colonnes requises
            missing = set(REQUIRED_COLUMNS) - set(X_renamed.columns)
            if missing:
                raise ValueError(f"Features manquantes: {missing}")
            
            # 3. Copie des données et remplacement simple des NaN
            X_clean = X_renamed.copy()
            
            # Remplacer tous les NaN par la médiane de chaque colonne
            for col in X_clean.columns:
                if X_clean[col].isnull().any():
                    median_val = X_clean[col].median()
                    X_clean[col] = X_clean[col].fillna(median_val)
                    logger.info(f"NaN dans {col} remplacés par médiane: {median_val:.2f}")
            
            # 4. Conversion sécurisée en tensor
            X_array = X_clean.values.astype(np.float32)
            X_tensor = torch.tensor(X_array, dtype=torch.float32, device=self.device)
            
            # 5. Vérification rapide
            if torch.isnan(X_tensor).any():
                raise RuntimeError("NaN détectés après conversion")
            
            # 6. Prédiction
            with torch.no_grad():
                raw_scores = self.model(X_tensor).cpu().numpy().flatten()
            
            
            # 6. Vérification des scores bruts
            if np.isnan(raw_scores).any():
                nan_count = np.isnan(raw_scores).sum()
                logger.error(f"NaN détectés dans les prédictions: {nan_count}/{len(raw_scores)} valeurs")
                raise RuntimeError("Le modèle a produit des scores NaN")
            
            # 7. Normalisation vers [0,1]
            normalized_scores = self._normalize_scores(raw_scores)
            
            # 8. Attribution des notes selon l'échelle fixe
            ratings = self._assign_ratings(normalized_scores)
            
            # 9. Formatage des résultats
            results = pd.DataFrame({
                'etf_id': X['etfId'].values,
                'raw_score': raw_scores,
                'normalized_score': normalized_scores,
                'rating': ratings
            })
            
            # 10. Validation des résultats
            self._validate_ratings(results)
            
            self.monitor.log_operation_success('prediction')
            return results
            
        except Exception as e:
            self.monitor.log_operation_failure('prediction', str(e))
            logger.exception("Échec lors de la prédiction")
            
            # Fallback: retourner un DataFrame vide avec la structure attendue
            return pd.DataFrame({
                'etf_id': [],
                'raw_score': [],
                'normalized_score': [],
                'rating': []
            })

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalise les scores entre 0 et 1"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        finite=scores[np.isfinite(scores)]

        if finite.size==0:
            return np.full_like(scores,0)
        
        if max_score - min_score < 1e-8:
            return np.zeros_like(scores) + 0.5
        
        normalized = (scores - min_score) / (max_score - min_score)
        
        # Ajustement pour éviter les erreurs d'arrondi
        return np.clip(normalized, 0.0, 1.0)
    
    def _assign_ratings(self, normalized_scores: np.ndarray) -> pd.Series:
        """Attribution des notes selon l'échelle fixe"""
        ratings = []
        for score in normalized_scores:
            # Gestion précise des bornes
            if score >= 0.9:
                ratings.append('A+')
            elif score >= 0.8:
                ratings.append('A')
            elif score >= 0.6:
                ratings.append('B')
            elif score >= 0.4:
                ratings.append('C')
            else:
                ratings.append('D')
        return pd.Series(ratings)
    
    def _validate_ratings(self, results: pd.DataFrame):
        """Validation de la cohérence des notes - Version Corrigée"""
        for _, row in results.iterrows():
            score = row['normalized_score']
            rating = row['rating']
            
            # Vérification que la note correspond au score
            min_score, max_score = self.RATING_SCALE[rating]
            
            # Gestion spéciale pour A+ (inclusion de la borne supérieure)
            if rating == 'A+':
                if not (0.9 <= score <= 1.0):
                    raise ValueError(
                        f"Incoherent rating {rating} for score {score:.4f}. "
                        f"Expected between 0.9 and 1.0 (inclusive)"
                    )
            else:
                if not (min_score <= score < max_score):
                    raise ValueError(
                        f"Incoherent rating {rating} for score {score:.4f}. "
                        f"Expected between {min_score} and {max_score}"
                    )