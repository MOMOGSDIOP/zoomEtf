#Fichiers des fonctionnalies 

def get_etf_data(isin):
    if cache.exists(isin):
        return cache.get(isin)
    else:
        data = yahoo_api.fetch(isin) or scraper.fetch(isin)
        cache.set(isin, data)
        return datass