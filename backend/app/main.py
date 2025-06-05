"""
Point d'entrée principal de l'API ZoomETF
"""


from fastapi import FastAPI
from app.api.endpoints import etfs
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="ZoomETF API",
    description="API Officielle du Bloomberg des ETFs",
    version="0.1.0"
)


# Routage
app.include_router(etfs.router)

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