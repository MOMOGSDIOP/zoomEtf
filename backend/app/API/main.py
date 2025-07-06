# backend/app/API/main.py
import os
import csv
import json
import redis
from fastapi import APIRouter, HTTPException, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.API.utils.alpha_vintage import fetch_enriched_etf_data
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

@router.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@router.get("/etf/{ticker}/full")
async def get_etf_full(ticker: str):
    try:
        cache_key = f"etf:{ticker.lower()}"
        if cached := redis_client.get(cache_key):
            return json.loads(cached)

        api_key = os.getenv("ALPHA_VANTAGE_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Clé API manquante")

        data = await fetch_enriched_etf_data(api_key, ticker)
        redis_client.setex(cache_key, 900, json.dumps(data))
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")


@router.get("/etfs/full")
async def get_all_etfs():
    try:
        api_key = os.getenv("ALPHA_VANTAGE_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Clé API manquante")

        # Chemin absolu pour le fichier CSV
        filepath = os.path.join(os.path.dirname(__file__), "..","data", "etf_list.csv")
        filepath = os.path.abspath(filepath)
        
        print(f"Trying to open CSV at: {filepath}")  # Debug
        
        if not os.path.exists(filepath):
            raise HTTPException(
                status_code=404,
                detail=f"Fichier etf_list.csv manquant à l'emplacement: {filepath}"
            )

        results = []
        with open(filepath, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if "symbol" not in reader.fieldnames:
                raise HTTPException(
                    status_code=500,
                    detail="Colonne 'symbol' manquante dans le CSV"
                )
                
            for row in reader:
                symbol = row["symbol"].strip()
                if not symbol:
                    continue
                    
                cache_key = f"etf:{symbol.lower()}"
                
                # Debug
                print(f"Processing ETF: {symbol}")

                if cached := redis_client.get(cache_key):
                    results.append(json.loads(cached))
                    continue

                try:
                    data = await fetch_enriched_etf_data(api_key, symbol)
                    redis_client.setex(cache_key, 900, json.dumps(data))
                    results.append(data)
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                    continue

        return results

    except Exception as e:
        print(f"Global error: {str(e)}")  # Debug
        raise HTTPException(
            status_code=500,
            detail=f"Erreur serveur: {str(e)}"
        )