"""
Point d'entrée principal de l'API ZoomETF
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import etfs, auth
from prometheus_fastapi_instrumentator import Instrumentator
from app.db.session import engine  # ton engine SQLAlchemy
from app.db.base_class import Base  # ta base SQLAlchemy où sont définis les modèles


app = FastAPI(
    title="ZoomETF API",
    description="API Officielle du Bloomberg des ETFs",
    version="0.1.0"
)

# Configuration CORS (à adapter en prod pour limiter l'origine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routage
app.include_router(etfs.router)
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])

@app.get("/health")
async def health_check():
    """Endpoint de vérification du statut"""
    return {"status": "OK", "service": "ZoomETF"}

# Exemple route
@app.get("/")
def root():
    return {"message": "Hello ZoomETF"}

# Active les métriques Prometheus
Instrumentator().instrument(app).expose(app)

