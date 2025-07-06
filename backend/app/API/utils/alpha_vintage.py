import httpx
from datetime import datetime

async def fetch_enriched_etf_data(api_key: str, ticker: str):
    async with httpx.AsyncClient() as client:
        # Time Series Data (fonctionne bien)
        ts_params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": api_key,
            "outputsize": "compact"
        }
        ts_response = await client.get("https://www.alphavantage.co/query", params=ts_params)
        ts_data = ts_response.json()

        if "Error Message" in ts_data or "Note" in ts_data:
            raise Exception(f"Alpha Vantage API error: {ts_data.get('Error Message', ts_data.get('Note'))}")

        # ETF Specific Data (utilisez un endpoint différent)
        etf_params = {
            "function": "OVERVIEW",
            "symbol": ticker,
            "apikey": api_key
        }
        etf_response = await client.get("https://www.alphavantage.co/query", params=etf_params)
        etf_data = etf_response.json()

        time_series = ts_data.get("Time Series (Daily)", {})
        if not time_series:
            raise Exception("No time series data available")

        latest_date = sorted(time_series.keys())[-1]
        latest = time_series[latest_date]

        # Données de base pour tous les ETFs
        base_data = {
            "symbol": ticker,
            "last_updated": datetime.now().isoformat(),
            "price_data": {
                "date": latest_date,
                "open": float(latest["1. open"]),
                "high": float(latest["2. high"]),
                "low": float(latest["3. low"]),
                "close": float(latest["4. close"]),
                "volume": int(latest["5. volume"]),
                "change": (float(latest["4. close"]) - float(latest["1. open"])) / float(latest["1. open"]) * 100
            }
        }

        # Données supplémentaires si disponibles
        if "Name" in etf_data:
            additional_data = {
                "name": etf_data.get("Name"),
                "description": etf_data.get("Description"),
                "sector": etf_data.get("Sector"),
                "asset_type": etf_data.get("AssetType"),
                "isin": etf_data.get("ISIN"),
                "market_cap": etf_data.get("MarketCapitalization")
            }
            return {**base_data, **additional_data}
        
        # Fallback pour les ETFs sans données supplémentaires
        return {
            **base_data,
            "name": f"ETF {ticker}",
            "asset_type": "ETF",
            "sector": "Diversifié"
        }