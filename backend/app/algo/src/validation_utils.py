"""
Utils de validation croisée pour ETFs
"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class ETFValidator:
    @staticmethod
    def time_series_cv(model, X, y, n_splits=5):
        """Validation croisée temporelle"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
            
        return np.mean(scores)
    
    @staticmethod
    def stress_test(model, X, stress_scenarios):
        """Test sur scénarios extrêmes"""
        results = {}
        for name, scenario in stress_scenarios.items():
            X_stress = scenario.apply(X)
            results[name] = model.predict(X_stress)
        return results


# Dans validation_utils.py
class WalkForwardTester:
    def test(self, model, data, n_splits=5):
        tscv = TimeSeriesSplit(n_splits)
        for train_idx, test_idx in tscv.split(data):
            train, test = data.iloc[train_idx], data.iloc[test_idx]
            model.fit(train)
            yield model.score(test)