import React from 'react';
import '../styles/ETFFilters.css';
import countries from '../data/countries'; // <- Utilisation correcte

const types = ['Accumulation', 'Capitalisation', 'Hybride'];
const replications = ['Physique', 'Synthétique'];
const sectors = [
  'Tous secteurs',
  'Technologie',
  'Santé',
  'Finance',
  'Énergie',
  'Consommation',
  'Immobilier',
  'Industrie',
];

const SortArrow = ({ order }) => (
  <span style={{ fontSize: '1.2rem', marginLeft: 4 }}>
    {order === 'asc' ? '↑' : '↓'}
  </span>
);

export default function ETFFilters({ filters, setFilters, onReset, onApply, onChangeSort, sortKey, sortOrder }) {
  function handleFilterChange(field, value) {
    setFilters({
      ...filters,
      [field]: value,
    });
  }

  function handleSortChange(field) {
    if (!onChangeSort) return;
    onChangeSort(field);
  }

  return (
    <div className="etf-filters-container">
      <div className="filters-header">
        <h3>Filtres ETFs</h3>
        <button className="btn-reset" onClick={onReset}>
          Réinitialiser
        </button>
      </div>

      <div className="filters-row">
        <div className="filter-block">
          <label>Région</label>
          <select value={filters.region} onChange={e => handleFilterChange('region', e.target.value)}>
            <option value="">-- Toutes --</option>
            {countries.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        <div className="filter-block">
          <label>Type</label>
          <select value={filters.type} onChange={e => handleFilterChange('type', e.target.value)}>
            <option value="">-- Tous --</option>
            {types.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        <div className="filter-block">
          <label>Réplication</label>
          <select value={filters.replication} onChange={e => handleFilterChange('replication', e.target.value)}>
            <option value="">-- Toutes --</option>
            {replications.map(r => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </div>

        <div className="filter-block">
          <label>Secteur</label>
          <select value={filters.sector} onChange={e => handleFilterChange('sector', e.target.value)}>
            {sectors.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <div className="filter-block">
          <label>Disponibilité</label>
          <select value={filters.availability} onChange={e => handleFilterChange('availability', e.target.value)}>
            <option value="">-- Toutes --</option>
            {countries.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="filters-actions">
        <button className="btn-apply" onClick={onApply} 
        style={{padding: '8px 16px',
      fontSize: '1rem',
      backgroundColor: '#1976d2',
      color: 'white',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',}}>OK</button>
      </div>

      <div className="sort-controls">
        <span
          className={sortKey === 'performance' ? 'sort-active' : 'sort-inactive'}
          onClick={() => handleSortChange('performance')}
        >
          Trier par Performance {sortKey === 'performance' ?  <SortArrow order={sortOrder} /> : ''}
        </span>
        <span
          className={sortKey === 'price' ? 'sort-active' : 'sort-inactive'}
          onClick={() => handleSortChange('price')}
        >
          Trier par Prix {sortKey === 'price' ?  <SortArrow order={sortOrder} />  : ''}
        </span>
      </div>
    </div>
  );
}
