"""
Configuration centrale du système
"""

# Catégories d'ETFs
DEFAULT_ETF_CATEGORIES = [
    'Equity', 'Fixed Income', 'Commodity', 
    'Alternative', 'Leveraged', 'ESG'
]

# Paramètres de risque
RISK_PARAMETERS = {
    'max_ter': 0.05,
    'min_liquidity': 1e6,
    'volatility_bounds': {
        'low': 0.05,
        'high': 0.5
    }
}

# Seuils de validation
VALIDATION_THRESHOLDS = {
    'price_deviation': 0.02,
    'tracking_error': 0.01,
    'required_fields': [
        'etfId', 'name', 'fundamentals.priceData.currentPrice'
    ]
}

# Configuration du modèle
MODEL_CONFIG = {
    'input_dim': 24,  # Doit correspondre au nombre exact de features générées
    'hidden_layers': [256, 128, 64],  # Réduit pour s'adapter à la dimension d'entrée
    'learning_rate': 1e-4,
    'dropout_rate': 0.3,
    'alpha': 0.7,  # Ajout des paramètres de loss
    'beta': 0.3,
    'batch_size': 64
}


GRAPH_CONFIG = {
    'normalize_features': True,
    'feature_specs': {
        'etf_features': [
            ('fundamentals.costs.ter', 'numeric'),
            ('fundamentals.liquidity.avgDailyVolume', 'numeric'),
            ('riskAnalysis.volatility.annualized', 'numeric')
        ],
        'asset_features': [
            ('weight', 'numeric'),
            ('contributionToTrackingError', 'numeric'),
            ('sector', 'categorical')
        ]
    }
}

#colonne  requis pour la configuration du système
REQUIRED_COLUMNS = [
    # Données de coûts
    "fundamentals.costs.ter",
    "fundamentals.costs.trackingError",
    
    # Données de liquidité
    "fundamentals.liquidity.avgDailyVolume",
    "fundamentals.liquidity.avgBidAskSpread",
    "fundamentals.liquidity.marketImpactScore",
    
    # Données de risque
    "riskAnalysis.volatility.30d",
    "riskAnalysis.drawdowns.maxDrawdown",
    "riskAnalysis.drawdowns.recoveryTimeDays",
    
    # Données alternatives
    "alternativeData.flows.30dNetFlow",
    "alternativeData.sentiment.newsSentiment",
    "alternativeData.sentiment.socialMediaSentiment",
    "alternativeData.sentiment.analystConsensus",
    
    # Facteurs
    "portfolio.characteristics.factorExposures.beta",
    "portfolio.characteristics.factorExposures.size",
    "portfolio.characteristics.factorExposures.value",
    "portfolio.characteristics.factorExposures.momentum",
    "portfolio.characteristics.factorExposures.quality",
    
    # Données techniques
    "replication.optimization.samplingError",
    "replication.lending.lendingRevenue",
    "alternativeData.ownership.institutionalPercentage",
    "replication.optimization.coverage",
    "riskAnalysis.liquidityRisk.basketLiquidityScore",
    
    # Métadonnées
    "metadata.creationDate"
]


VALIDATION_THRESHOLDS = {
    'required_fields': REQUIRED_COLUMNS,
}