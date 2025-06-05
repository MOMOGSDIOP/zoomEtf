import React from 'react';
import '../styles/UserPortfolio.css';

export default function UserPortfolio({ onNavigate }) {
  return (
    <div className="portfolio-home">
      <h2>Bienvenue dans votre portefeuille</h2>
      <div className="portfolio-columns">
        <div className="portfolio-section">
          <ul className="feature-list">
            <li onClick={() => onNavigate('list')}>📊 Liste de mes ETFs</li>
            <li onClick={() => onNavigate('performance')}>📈 Performance globale</li>
            <li onClick={() => onNavigate('analysis')}>🧠 Outils d'analyse</li>
            <li onClick={() => onNavigate('transaction')}>💼 Historique des transactions</li>
            <li onClick={() => onNavigate('alert')}>🔔 Alertes personnalisées</li>
          </ul>
        </div>

        <div className="portfolio-section">
          <ul className="feature-list">
            <li onClick={() => onNavigate('rebalancement')}>🔄 Rebalancement automatique</li>
            <li onClick={() => onNavigate('dividendes')}>📅 Calendrier de dividendes</li>
            <li onClick={() => onNavigate('export')}>📤 Export PDF/Excel</li>
            <li onClick={() => onNavigate('conseils')}>💬 Conseils intelligents</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
