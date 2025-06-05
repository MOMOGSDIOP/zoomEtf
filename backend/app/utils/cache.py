# backend/app/utils/cache.py

import redis
import pickle
from datetime import datetime

from app.core.settings import settings  # <- Import correct de l'objet settings

class CacheManager:
    def __init__(self):
        self.client = redis.Redis.from_url(settings.REDIS_URL)

    def get(self, key: str):
        """Récupère une valeur du cache"""
        if cached := self.client.get(key):
            return pickle.loads(cached)
        return None

    def set(self, key: str, value: any, ttl: int = 3600):
        """Stocke une valeur avec expiration"""
        self.client.setex(key, ttl, pickle.dumps(value))

    async def log_error(self, message: str):
        """Log une erreur dans Redis"""
        error_key = f"error:{datetime.utcnow().timestamp()}"
        self.client.hset(error_key, mapping={
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })

    def get_volatility(self, symbol: str) -> float:
        """Récupère la volatilité depuis Redis"""
        return float(self.client.get(f"volatility:{symbol}") or 1.0)


# Création du cache global
cache = CacheManager()
