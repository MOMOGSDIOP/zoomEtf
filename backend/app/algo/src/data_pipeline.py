"""
Pipeline avancé de traitement des données ETFs (corrigé et optimisé)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class ETFDataPipeline:
    
    def __init__(self):
        self.quantile_transformer = QuantileTransformer(output_distribution='normal',nquantile=10)
        self.imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=42)
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # Pour traquer les colonnes
        self.cat_cols = []
        self.num_cols = []

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline principal : aplatissement, nettoyage, encodage, normalisation"""
        df = self.flatten_nested_structures(df)  # Changement de nom pour plus de clarté
        df = self.add_temporal_features(df)
        df = self.handle_missing_and_encoding(df)
        df = self.process_numerical_features(df)
        return df

    def flatten_nested_structures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplatissement récursif des colonnes de type dict et list"""
        # D'abord traiter les dictionnaires
        dict_cols = df.select_dtypes(include='object').map(lambda x: isinstance(x, dict)).any()
        while dict_cols.any():
            for col in dict_cols[dict_cols].index:
                expanded = pd.json_normalize(df[col])
                expanded.columns = [f"{col}.{subcol}" for subcol in expanded.columns]
                df = df.drop(columns=col).join(expanded)
            dict_cols = df.select_dtypes(include='object').map(lambda x: isinstance(x, dict)).any()
        
        # Ensuite traiter les listes
        list_cols = df.map(lambda x: isinstance(x, list)).any()
        for col in list_cols[list_cols].index:
            # Convertir les listes en string ou les exploser selon le besoin
            df[col] = df[col].apply(lambda x: '|'.join(map(str, x)) if isinstance(x, list) else x)
        
        return df

    def handle_missing_and_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputation + encodage catégoriel avec vérification des types"""
        # Séparation numérique / catégoriel avec vérification
        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = [
            col for col in df.select_dtypes(exclude=np.number).columns
            if not df[col].map(lambda x: isinstance(x, (dict, list))).any()
        ]
        
        # Imputation numérique
        if self.num_cols:
            df[self.num_cols] = self.imputer.fit_transform(df[self.num_cols])
        
        # Imputation et encodage catégoriel
        if self.cat_cols:
            df[self.cat_cols] = df[self.cat_cols].fillna("Unknown")
            
            # Conversion en string pour garantir le bon fonctionnement de l'encodeur
            df[self.cat_cols] = df[self.cat_cols].astype(str)
            
            df[self.cat_cols] = self.encoder.fit_transform(df[self.cat_cols])
        
        return df

    def process_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalisation robuste pour toutes les colonnes numériques"""
        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if self.num_cols:
            # Handle negative/zero values safely before log1p
            df[self.num_cols] = df[self.num_cols].apply(
            lambda col: np.log1p(col - col.min() + 1e-6) if (col <= 0).any() else np.log1p(col))
            # Convert to float64 to avoid float32 issues later
            df[self.num_cols] = df[self.num_cols].astype(np.float64)
             # Apply quantile transformation
            df[self.num_cols] = self.quantile_transformer.fit_transform(df[self.num_cols])
        
        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajout de variables temporelles (si présentes)"""
        if 'date' in df.columns:
            df['days_since_inception'] = (pd.to_datetime('today') - pd.to_datetime(df['date'])).dt.days
            if 'returns' in df.columns:
                df['market_regime'] = df['returns'].rolling(90, min_periods=1).apply(self._detect_regime)
        return df

    def _detect_regime(self, returns: pd.Series) -> int:
        """Détecte bull/bear/neutral market"""
        if len(returns) < 2:  # Minimum 2 points pour calculer std et mean
            return 0
        vol = returns.std()
        mean = returns.mean()
        if vol > 0.25 and mean < -0.1:
            return -1
        elif vol < 0.15 and mean > 0.1:
            return 1
        return 0