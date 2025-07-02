"""
Scénarios de stress test pour les ETFs
"""

class ETFStressTester:
    def __init__(self, scenarios: List[Dict]):
        self.scenarios = self._load_scenarios(scenarios)
    
    def _load_scenarios(self, scenario_defs):
        """Initialise les scénarios à partir de la configuration"""
        return [
            {
                'name': s['name'],
                'apply': self._create_scenario_function(s)
            } 
            for s in scenario_defs
        ]
    
    def _create_scenario_function(self, scenario):
        """Crée dynamiquement des fonctions de scénario"""
        if scenario['type'] == 'market_crash':
            return lambda df: self._apply_market_crash(df, scenario['severity'])
        elif scenario['type'] == 'liquidity_shock':
            return lambda df: self._apply_liquidity_shock(df, scenario['factor'])
        else:
            return lambda df: df  # Scenario neutre par défaut
    
    def run_all_scenarios(self, data, model):
        """Exécute tous les scénarios sur les données"""
        results = {}
        for scenario in self.scenarios:
            try:
                scenario_data = scenario['apply'](data.copy())
                predictions = model.predict(scenario_data)
                results[scenario['name']] = {
                    'avg_score': np.mean(predictions),
                    'min_score': np.min(predictions),
                    'affected_etfs': self._identify_impacted(data, predictions)
                }
            except Exception as e:
                logger.error(f"Scenario {scenario['name']} failed: {str(e)}")
                continue
        return results
    
    def _apply_market_crash(self, data, severity):
        """Simule un krach boursier"""
        crash_factor = 1 - severity
        data['fundamentals.priceData.currentPrice'] *= crash_factor
        data['riskAnalysis.volatility.annualized'] *= (1 + severity)
        return data
    
    def _apply_liquidity_shock(self, data, factor):
        """Simule un choc de liquidité"""
        data['fundamentals.liquidity.avgDailyVolume'] /= factor
        data['fundamentals.liquidity.marketImpactScore'] *= factor
        return data
    
    def _identify_impacted(self, original_data, stressed_scores):
        """Identifie les ETFs les plus affectés"""
        delta = original_data['score'] - stressed_scores
        return delta.nlargest(5).to_dict()