# backend/app/core/config.py

import os

# Pour des chemins internes ou autres constantes de runtime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_NAME = "ZoomETF"
API_PREFIX = "/api"

# Feature flags statiques
ENABLE_CACHE = True
ENABLE_METRICS = True
