const etfs = [
{
  "etfId": "SPY-US-USD",
  "name": "SPDR S&P 500 ETF Trust",
  "metadata": {
    "issuer": "State Street Global Advisors",
    "description": "Seeks to provide investment results that correspond to the S&P 500 Index",
    "creationDate": "1993-01-22",
    "lastRebalanceDate": "2023-12-15"
  },
  
  // Données fondamentales enrichies
  "fundamentals": {
    "priceData": {
      "currentPrice": 422.50,
      "previousClose": 420.00,
      "bidAskSpread": 0.02,
      "nav": 422.30,
      "premiumDiscount": 0.05,
      "52WeekHigh": 432.10,
      "52WeekLow": 380.25
    },
    
    "costs": {
      "ter": 0.0009,
      "trackingDifference": -0.0012,
      "trackingError": 0.0021,
      "lendingRevenue": 0.0005,
      "transactionCosts": 0.0003
    },
    
    "liquidity": {
      "avgDailyVolume": 5000000,
      "avgBidAskSpread": 0.02,
      "impliedLiquidity": 15000000,
      "marketImpactScore": 0.12
    }
  },

  // Structure temporelle complète
  "timeSeries": {
    "dailyReturns": [0.003, -0.012, 0.008], // 252 derniers jours
    "historicalNav": [
      {"date": "2023-01-03", "value": 382.45},
      {"date": "2023-01-04", "value": 384.12},
      {"date": "2023-01-05", "value": 385.67}
    ],
    "trackingDifferenceHistory": [
      {"date": "2023-01", "value": -0.0015},
      {"date": "2023-02", "value": -0.0011},
      // Derniers mois
    ]
  },

  // Données de portefeuille détaillées
  "portfolio": {
    "holdings": [
      {
        "assetId": "AAPL-US",
        "name": "Apple Inc.",
        "weight": 0.072,
        "sector": "Technology",
        "country": "US",
        "contributionToTrackingError": 0.0012
      },
     // Tous les constituants
    ],
    
    "characteristics": {
      "sectorWeights": {
        "Technology": 0.28,
        "Health Care": 0.14,
        "Financials": 0.13,
      },
      "countryWeights": {
        "US": 1.00
      },
      "factorExposures": {
        "beta": 1.02,
        "size": 0.95,
        "value": -0.32,
        "momentum": 0.15,
        "quality": 0.78
      }
    }
  },

  // Métriques de risque avancées
  "riskAnalysis": {
    "volatility": {
      "30d": 0.138,
      "90d": 0.140,
      "annualized": 0.142
    },
    "drawdowns": {
      "maxDrawdown": -0.238,
      "avgDrawdown": -0.052,
      "recoveryTimeDays": 63
    },
    "correlations": {
      "spx": 0.998,
      "agg": 0.124,
      "gld": -0.034
    },
    "liquidityRisk": {
      "redemptionCost": 0.0018,
      "basketLiquidityScore": 0.92
    }
  },

  // Données alternatives
  "alternativeData": {
    "sentiment": {
      "newsSentiment": 0.72,
      "socialMediaSentiment": 0.65,
      "analystConsensus": 0.81
    },
    "flows": {
      "30dNetFlow": 1500000000,
      "ytdFlow": 12000000000
    },
    "ownership": {
      "institutionalPercentage": 0.62,
      "topHolder": "Vanguard Group"
    }
  },

  // Mécanisme de réplication
  "replication": {
    "method": "Full Replication",
    "optimization": {
      "samplingError": 0.0002,
      "numHoldings": 503,
      "coverage": 1.00
    },
    "lending": {
      "isLending": true,
      "lendingRevenue": 0.0005,
      "collateralQuality": "AAA"
    }
  },

  // Données comparatives
  "peerComparison": {
    "avgTer": 0.0012,
    "avgTrackingError": 0.0025,
    "percentileRank": {
      "cost": 0.95,
      "liquidity": 0.98,
      "tracking": 0.91
    }
  },

  // Metadata système
  "system": {
    "lastUpdated": "2023-12-20T15:30:00Z",
    "dataSource": "Bloomberg/State Street",
    "calculationMethodology": "MSCI RiskMetrics"
  }
}

];

export default etfs;