"""
ETFs Analysis System - Core Rating Model (Version Intégrée)
"""

import datetime
import json
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap
from datetime import datetime


# Import des modules internes
from advanced_loss import ETFCompositeLoss
from semi_supervised_model import ETFSemiSupervisedModel
from gnn_model import ETFGraphModel
from validation_utils import ETFValidator
from monitoring import ETFSystemMonitor
from data_pipeline import ETFDataPipeline
from stress_scenarios import ETFStressTester
from etf_feature_builder import ETFFeatureBuilder
from etf_graph import ETFGraphProcessor, ETFGraphConfig
from explanations import ETFExplanationGenerator
from memory_optimizer import MemoryOptimizer
from data_utils import DataPreprocessor
import data_utils as du

from config import MODEL_CONFIG, RISK_PARAMETERS, VALIDATION_THRESHOLDS, REQUIRED_COLUMNS



logger = logging.getLogger(__name__)

class ETFScoringEngine:
    """Moteur de notation unifié intégrant tous les composants"""
    
    def __init__(self, config: Dict):
        """Initialise le moteur de notation complet
          Args:
          config: Configuration du système (voir config.py)"""
        
        self.config = config
        self.device = self._init_device()
        self.monitor = ETFSystemMonitor()
        self.data_pipeline = ETFDataPipeline()
        self.stress_tester = ETFStressTester(config['stress_scenarios'])
        graph_config = ETFGraphConfig(
            normalize_features=True,
            use_alternative_data=config.get('USE_ALTERNATIVE_DATA', False),
            use_temporal_features=config.get('USE_TEMPORAL_FEATURES', False),
            device=self.device
        )

    
        self.graph_processor = ETFGraphProcessor(graph_config)

        # Initialisation avec la configuration complète
        self.feature_builder = ETFFeatureBuilder({
            'RISK_PARAMETERS': config.get('RISK_PARAMETERS', {}),
            'VALIDATION_THRESHOLDS': config.get('VALIDATION_THRESHOLDS', {})
        })
        
        # Vérification de cohérence dimensionnelle
        """if len(self.feature_builder.transform(pd.DataFrame(columns=REQUIRED_COLUMNS)).columns) != config['MODEL_CONFIG']['input_dim']:
            raise ValueError("Incompatibilité dimension features/modèle")"""
        
        
        # Initialisation des modèles
        self._init_models()
        self._init_loss_and_optim()
        self.explanation_generator = ETFExplanationGenerator(model=self.semi_supervised_model,device=self.device)
        self.memory_optimizer = MemoryOptimizer(device=self.device, safety_factor=0.7)
        
        logger.info("ETFScoringEngine initialized with config: %s", config)
    
    def _init_device(self) -> torch.device:
        """Détermine le device de calcul optimal"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _init_models(self):
        """Initialise les différents modèles composants"""
        # Modèle semi-supervisé principal
        self.semi_supervised_model = ETFSemiSupervisedModel(
            self.config['input_dim']
        ).to(self.device)
        
        # Modèle graphique (si données disponibles)
        self.gnn_model = ETFGraphModel(
            self.config['gnn_input_dim']
        ).to(self.device)
        
        # Explicateur SHAP
        self.explainer = None
    
    def _init_loss_and_optim(self):
        """Initialise les fonctions de loss et optimiseurs"""
        self.loss_fn = ETFCompositeLoss(
            alpha=self.config['alpha'],
            beta=self.config['beta']
        )
        
        params = list(self.semi_supervised_model.parameters()) + \
                 list(self.gnn_model.parameters())
                 
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
    
    
    def train(self, X: pd.DataFrame, y: pd.Series, graph_data: Optional[Dict] = None,epochs: int = 100) -> Dict:
        """Entraînement complet du système intégrant modèle semi-supervisé et GNN
           Args:
           X: DataFrame contenant les features d'entraînement
           y: Series contenant les cibles supervisées
           graph_data: Données graphiques optionnelles pour le GNN (format brut ou prétraité)
           epochs: Nombre d'époques d'entraînement (défaut: 100)    
         Returns:
             Dict: Métriques consolidées d'entraînement contenant:
            - supervised_loss: Perte supervisée
            - vae_loss: Perte du VAE
            - gnn_loss: Perte du GNN (si applicable)
            - correlation: Corrélation du GNN (si applicable)
        Raises:
        ValueError: Si les données d'entrée sont invalides
        RuntimeError: Si l'entraînement échoue pour des raisons techniques"""
        # Début du monitoring
        
        self.monitor.log_operation_start('full_training')
        self.monitor.track_performance('training_epochs', epochs)

        try:
            # Validation des entrée
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X doit être un DataFrame pandas")
            
            if not isinstance(y, pd.Series):
                raise ValueError("y doit être une Series pandas")
            
            if graph_data is not None and not isinstance(graph_data, dict):
                raise ValueError("graph_data doit être un dictionnaire si fourni")
            
            if len(X) != len(y):
                raise ValueError("X et y doivent avoir la même longueur")
            if epochs <= 0:
                raise ValueError("Le nombre d'époques doit être positif")
            
            # 1. Prétraitement des données
            X_processed = self.data_pipeline.process(X)
            if X_processed.empty:
                raise ValueError("Le DataFrame prétraité est vide")

            X_tensor, y_tensor = self._prepare_tensors(X_processed, y)
            
            # 2. Entraînement semi-supervisé principal
            train_metrics = self._train_semi_supervised(X_tensor, y_tensor, epochs)
            self.monitor.track_performance('supervised_loss', train_metrics['supervised_loss'])
            
            # 3. Fine-tuning avec GNN si données disponibles
            if graph_data is not None:
                gnn_metrics = self._train_with_gnn(graph_data, y_tensor)
                train_metrics.update(gnn_metrics)
                self.monitor.track_performance('gnn_loss', gnn_metrics['gnn_loss'])
                
            # 4. Initialisation de l'explicateur SHAP
            self._init_explainer(X_processed)
            
            # Suivi des performances globales
            self.monitor.track_performance('total_training_loss',train_metrics.get('gnn_loss', train_metrics['supervised_loss']))
            self.monitor.log_operation_success('full_training')
            
            return train_metrics
        
        except ValueError as ve:
            error_msg = f"Validation error in training: {str(ve)}"
            self.monitor.log_operation_failure('full_training', error_msg)
            logger.error(error_msg)
            raise
        
        except RuntimeError as re:
            error_msg = f"Training execution error: {str(re)}"
            self.monitor.log_operation_failure('full_training', error_msg)
            logger.error(error_msg)
            raise
        
        except Exception as e:
            error_msg = f"Unexpected training error: {str(e)}"
            self.monitor.log_operation_failure('full_training', error_msg)
            logger.error(error_msg, exc_info=True)
            raise
         
    
    
    def _train_semi_supervised(self, X: torch.Tensor, y: torch.Tensor, 
                              epochs: int) -> Dict:
        """Entraînement du modèle semi-supervisé"""
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                try:
                    self.optimizer.zero_grad()
                    # Forward pass supervisé
                    preds = self.semi_supervised_model(batch_X)
                    supervised_loss = self.loss_fn(preds, batch_y)
                    # Forward pass VAE
                    recon, mu, logvar = self.semi_supervised_model(batch_X, supervised=False)
                    vae_loss = self.loss_fn(preds, batch_y, (recon, mu, logvar))
                    # Backpropagation
                    total_loss = supervised_loss + vae_loss
                    total_loss.backward()
                    self.optimizer.step()
                
                finally:
                    self.memory_optimizer.clear_tensors(batch_X, batch_y, preds, recon, mu, logvar)
        
        return {
            'supervised_loss': supervised_loss.item(),
            'vae_loss': vae_loss.item()}

    

    def _train_with_gnn(self, graph_data: Dict, y: torch.Tensor) -> Dict:
        """Effectue l'entraînement avec le modèle GNN en utilisant le monitoring système
         Args:graph_data: Données graphiques sous forme brute ou prétraitées y: 
         Tensors des cibles d'entraînement"""

        self.monitor.log_operation_start('gnn_training')
        try:
            # Validation de y
            if y is None or not isinstance(y, torch.Tensor):
                raise ValueError("Target y must be a valid torch.Tensor")
            
            # Conversion ou validation des données graph
            if not isinstance(graph_data, dict) or 'x' not in graph_data:
                self.monitor.track_performance('data_conversion', 1)
                graph_data = self._get_graph_data(graph_data)
            
            else:
                self.monitor.track_performance('data_conversion', 0)
            
            # Appel direct à la méthode d'entraînement du modèle GNN
            metrics = self.gnn_model.train(
                graph_data=graph_data,
                targets=y,
                downstream_model=self.semi_supervised_model,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                device=self.device)
            
            # Suivi
            self.monitor.track_performance('gnn_training', metrics['gnn_loss'])
            self.monitor.log_operation_success('gnn_training')
            
            return metrics
        
        except ValueError as ve:
            msg = f"Validation error in GNN training: {str(ve)}"
            self.monitor.log_operation_failure('gnn_training', msg)
            logger.error(msg)
            raise
        
        except RuntimeError as re:
            msg = f"Runtime error during GNN training: {str(re)}"
            self.monitor.log_operation_failure('gnn_training', msg)
            logger.error(msg)
            raise
        
        except Exception as e:
            msg = f"Unexpected error in GNN training: {str(e)}"
            self.monitor.log_operation_failure('gnn_training', msg)
            logger.error(msg, exc_info=True)
            raise


    def _prepare_etf_features(self, etf_data: pd.DataFrame) -> pd.DataFrame:
        """Transforme les données ETF brutes en features pour le modèle"""
        
        return self.feature_builder.transform(etf_data)
    
    def _validate_config(self):
        """Vérifie la cohérence de la configuration du moteur"""
        required_keys = ['input_dim', 'hidden_layers', 'learning_rate', 
                   'gnn_input_dim', 'REQUIRED_COLUMNS']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Configuration manquante: {key}")

    
    def _generate_explanations(self, etf_data: pd.DataFrame, scores: np.ndarray) -> Dict:
        features = self._prepare_etf_features(etf_data)
        return self.explanation_generator.generate_explanations(
            etf_data=etf_data,
            scores=scores,
            prepared_features=features)
    
    
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prédit les notes pour les ETFs avec explications"""
        self.monitor.log_operation_start('prediction')
        
        try:
            # 1. Préparation des features
            features = self._prepare_etf_features(X)
            
            # 2. Prédiction du modèle
            with torch.no_grad():
                X_tensor = torch.FloatTensor(features.values).to(self.device)
                raw_scores = self.semi_supervised_model(X_tensor).cpu().numpy()
            
            # 3. Conversion en note (A+ à D)
            ratings = pd.cut(
                raw_scores.flatten(),
                bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
                labels=['D', 'C', 'B', 'A', 'A+']
            )
            
            # 4. Formatage des résultats
            results = pd.DataFrame({
                'etf_id': X['etfId'],
                'raw_score': raw_scores.flatten(),
                'rating': ratings,
                **self._generate_explanations(X, raw_scores)
            })
            
            self.monitor.log_operation_success('prediction')
            return results
            
        except Exception as e:
            self.monitor.log_operation_failure('prediction', str(e))
            raise

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        X_tensor, y_tensor = self._prepare_tensors(X, y)
        with torch.no_grad():
            preds = self.semi_supervised_model(X_tensor)
            mse = nn.MSELoss()(preds, y_tensor).item()
            mae = torch.mean(torch.abs(preds - y_tensor)).item()
            
        return {'MSE': mse, 'MAE': mae}

        
    
    def explain(self, X: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
        """
        Génère des explications SHAP pour les prédictions
        
        Args:
            X: Données à expliquer
            sample_size: Taille de l'échantillon pour approximation
            
        Returns:
            DataFrame avec les valeurs SHAP
        """
        if self.explainer is None:
            self._init_explainer(X)
            
        X_processed = self.data_pipeline.process(X)
        sample = X_processed.sample(min(sample_size, len(X)))
        
        shap_values = self.explainer.shap_values(sample)
        return pd.DataFrame(
            shap_values,
            columns=X_processed.columns,
            index=sample.index
        )
    
    def evaluate_stress_scenarios(self, X: pd.DataFrame) -> Dict:
        """
        Évalue les performances sous différents scénarios de stress
        
        Args:
            X: Données de base
            
        Returns:
            Résultats des stress tests
        """
        return self.stress_tester.run_all_scenarios(X, self)
    
    def save(self, path: str):
        """Sauvegarde l'état complet du moteur"""
        state = {
            'model_state': self.semi_supervised_model.state_dict(),
            'gnn_state': self.gnn_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(state, path)
        logger.info("Model saved to %s", path)
    
    def load(self, path: str):
        """Charge un état sauvegardé"""
        state = torch.load(path)
        self.semi_supervised_model.load_state_dict(state['model_state'])
        self.gnn_model.load_state_dict(state['gnn_state'])
        self.optimizer.load_state_dict(state['optimizer'])
        logger.info("Model loaded from %s", path)
        
    def run_full_analysis(self, raw_etf_data: List[Dict]) -> Dict:
        """Pipeline complet: données → notation → analyse"""
        try:
            # 1. Conversion des données JSON
            etf_df = pd.json_normalize(raw_etf_data)
            
            # 2. Prétraitement
            processed_data = self.data_pipeline.process(etf_df)
            
            # 3. Notation
            ratings = self.predict(processed_data)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'ratings': ratings.to_dict('records'),
                'data_stats': {
                    'num_etfs': len(ratings),
                    'avg_score': ratings['raw_score'].mean()
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        
    
    
    ## LEs méthodes privées pour la préparation des données et des tensors
    # sont utilisées pour transformer les données brutes en formats utilisables par les modèles.
    

    def _prepare_tensors(self, X: pd.DataFrame, y: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        
        try:

            if X.isnull().values.any():
                raise ValueError("NaN detected in features")
                
            if y.isnull().values.any():
                raise ValueError("NaN detected in target")
            
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            
            if isinstance(y, (pd.Series, np.ndarray)):
                y_tensor = torch.FloatTensor(y.values).to(self.device)
            else:
                y_tensor = torch.FloatTensor(y).to(self.device)
            
            if X_tensor.shape[0] != y_tensor.shape[0]:
                raise ValueError("Mismatch in number of samples between X and y")
            
            return X_tensor, y_tensor
        
        except Exception as e:
            logger.error(f"Tensor conversion failed: {str(e)}")
            raise


    def _prepare_graph_data(self, etf_data: List[Dict]) -> Dict:
        """Alias vers la nouvelle implémentation"""
        return self.graph_processor.prepare_graph_data(etf_data) 


    def _init_explainer(self, etf_data: pd.DataFrame) -> None:
        """Initialise SHAP en exploitant pleinement le dataset ETF.
         Args:
         etf_data: DataFrame issu du JSON avec colonnes:
            - 'fundamentals.liquidity.avgDailyVolume' (obligatoire)
            - 'etfId' (pour le déduplication)
            - Toutes colonnes requises par _prepare_etf_features()"""
            
        #Vérification renforcée du schéma de donnée
        required_columns = {
            'fundamentals.liquidity.avgDailyVolume',
            'etfId'}

        missing = required_columns - set(etf_data.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        try:
            # 1. Sélection des ETFs représentatifs
            background = (
                etf_data
                .sort_values('fundamentals.liquidity.avgDailyVolume', ascending=False)
                .drop_duplicates('etfId')
                .head(50)  # Taille fixe pour la stabilité
                )
            
            logger.debug(f"SHAP background: {len(background)} ETFs sélectionnés")
            # 2. Vérification du contenu des features
            if 'fundamentals.costs.ter' not in background.columns:
                logger.warning("TER manquant - utilisation de 0.001 comme valeur par défaut")
                background['fundamentals.costs.ter'] = background.get('fundamentals.costs.ter', 0.001)
            
            # 3. Préparation des features
            features = self._prepare_etf_features(background)
            if features.isnull().any().any():
                raise ValueError("NaN détectés dans les features après préparation")

            # 4. Conversion pour SHAP
            background_tensor = torch.FloatTensor(features.values).to(self.device)

            # 5. Initialisation avec vérification du modèle
            self.explainer = shap.DeepExplainer(
                model=self.semi_supervised_model,
                data=background_tensor)
            
            logger.info(f"SHAP initialisé. Shape: {background_tensor.shape}")
        
        except Exception as e:
            logger.error("Échec initialisation SHAP", exc_info=True)
            raise RuntimeError(f"Erreur SHAP: {str(e)}") from e


    def _get_graph_data(self, etf_data: List[Dict]) -> Dict:
        """Alias vers la nouvelle implémentation"""
        return self.graph_processor.prepare_graph_data(etf_data)
    




if __name__ == "__main__":
    from config import MODEL_CONFIG, RISK_PARAMETERS, VALIDATION_THRESHOLDS, REQUIRED_COLUMNS
    data_preprocessor = DataPreprocessor()

    config = {
        # Paramètres modèle (tirés directement de MODEL_CONFIG)
        'MODEL_CONFIG': MODEL_CONFIG,
        'input_dim': MODEL_CONFIG['input_dim'],
        'hidden_layers': MODEL_CONFIG['hidden_layers'],
        'learning_rate': MODEL_CONFIG['learning_rate'],
        'dropout_rate': MODEL_CONFIG['dropout_rate'],
        'alpha': MODEL_CONFIG['alpha'],
        'beta': MODEL_CONFIG['beta'],
        'batch_size': MODEL_CONFIG['batch_size'],
        'weight_decay': 1e-5,  # Ajout manquant dans MODEL_CONFIG
        
        # Configuration GNN (ajout pour cohérence)
        'gnn_input_dim': 3,  # Correspond aux features des noeuds dans _get_graph_data
        
        # Paramètres de risque
        'RISK_PARAMETERS': RISK_PARAMETERS,
        'VALIDATION_THRESHOLDS': VALIDATION_THRESHOLDS,
        
        # Scénarios de test (maintenant alignés avec RISK_PARAMETERS)
        'stress_scenarios': [
            {
                'name': 'Market Crash', 
                'type': 'market_crash', 
                'severity': RISK_PARAMETERS['volatility_bounds']['high'] * 1.5
            },
            {
                'name': 'Liquidity Shock', 
                'type': 'liquidity_shock', 
                'factor': RISK_PARAMETERS['min_liquidity'] * 0.5
            }
        ],
        
        # Colonnes requises
        'REQUIRED_COLUMNS': REQUIRED_COLUMNS
    }
    
    # 2. Initialisation avec vérification
    
    try:
        engine = ETFScoringEngine(config)
        logger.info("Engine initialized successfully")
        
        # Chargement des données
        with open('etf_data_test.json') as f:
            etf_data = json.load(f)
            
        # Vérification des champs requis
        sample_item = etf_data[0]
        missing_fields = [field for field in REQUIRED_COLUMNS 
                        if not du._nested_field_exists(sample_item, field)]
        
        if missing_fields:
            logger.warning("Missing required fields: %s", missing_fields)
        
        # Traitement des données
        processed_data = data_preprocessor.process_numerical_data(
            pd.DataFrame(etf_data)
        )
        
        # Exécution et sauvegarde
        results = engine.run_full_analysis(processed_data.to_dict('records'))
        du.save_results(results)  # Utilisation de notre fonction sécurisée
      
    except Exception as e:
        logger.error("System failure: %s", str(e), exc_info=True)
        raise