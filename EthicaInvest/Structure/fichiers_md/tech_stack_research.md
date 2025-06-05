# Recherche de la Stack Technique

## APIs à Explorer pour la Phase 2
1. **Yahoo Finance API** - Données marché temps réel
2. **EODHistoricalData** - Données historiques
3. **Morningstar API** - Données fondamentales
4. **Bloomberg API** (si accès possible)
5. **Refinitiv API** (anciennement Thomson Reuters)

## Outils de Scraping (si APIs limitées)
- Python: BeautifulSoup, Selenium
- Node.js: Puppeteer

## Base de Données
- PostgreSQL (relations complexes entre ETFs)
- Redis (pour les données temps réel)

## Architecture Backend
- Options:
  - FastAPI (Python) + SQLAlchemy
  - Express.js (Node.js) + Prisma ORM

## Frontend
- React.js + Tailwind CSS
- Librairies de visualisation:
  - Recharts (graphiques simples)
  - D3.js (visualisations complexes)
  - TradingView widget (graphiques avancés)

## IA à Intégrer
- OpenAI GPT-4-turbo pour:
  - Résumés automatiques d'ETFs
  - Recommandations personnalisées
  - Chatbot expert ETF
- Modèles de scoring personnalisés:
  - Par secteur
  - Volatilité
  - Performance ajustée au risque