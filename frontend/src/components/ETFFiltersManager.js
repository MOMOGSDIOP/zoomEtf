import React, { useState, useMemo } from 'react';
import ETFFilters from './ETFFilters';
import ETFTable from './ETFTable';
import etfs from '../data/etfs';
import '../styles/ETFFilters.css';

export default function ETFFiltersManager({ type, onBack, onSelectETF }) {
  const [filters, setFilters] = useState({
    region: '',
    type: '',
    replication: '',
    sector: '',
    availability: '',
  });
  const [appliedFilters, setAppliedFilters] = useState(filters);
  const [sortKey, setSortKey] = useState('performance');
  const [sortOrder, setSortOrder] = useState('desc');

  const onChangeSort = (key) => {
    if (key === sortKey) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortOrder(key === 'performance' ? 'desc' : 'asc');
    }
  };

  const resetFilters = () => {
    setFilters({
      region: '',
      type: '',
      replication: '',
      sector: '',
      availability: '',
    });
    setAppliedFilters({
      region: '',
      type: '',
      replication: '',
      sector: '',
      availability: '',
    });
  };

  const applyFilters = () => {
    setAppliedFilters(filters);
  };

  const filteredEtfs = useMemo(() => {
    return etfs
      .filter((e) => {
        const { region, type, replication, sector, availability } = appliedFilters;

        // Filtre region & availability combiné selon logique précise
        if (availability && availability !== '') {
          if (availability === 'Partout') {
            // Seulement les ETFs disponibles partout
            if (e.availability !== 'Partout') return false;
          } else {
            // Filtre sur une région précise
            // On garde ceux disponibles dans la région OU partout
            if (e.availability !== availability && e.availability !== 'Partout') return false;
          }
        }

        if (region && region !== '') {
          // On veut un filtre region strict si renseigné
          // Pour éviter conflit avec availability, on filtre ici sur region si disponible
          if (e.region !== region) return false;
        }

        // Autres filtres classiques
        if (type && type !== '') {
          if (e.type !== type) return false;
        }

        if (replication && replication !== '') {
          if (e.replication !== replication) return false;
        }

        if (sector && sector !== '') {
          if (sector !== 'Tous secteurs' && e.sector !== sector) return false;
        }

        return true;
      })
      .sort((a, b) => {
        if (sortKey === 'performance') {
          return sortOrder === 'asc'
            ? a.performance - b.performance
            : b.performance - a.performance;
        }
        if (sortKey === 'price') {
          return sortOrder === 'asc' ? a.price - b.price : b.price - a.price;
        }
        return 0;
      });
  }, [appliedFilters, sortKey, sortOrder]);

  return (
    <>
      <ETFFilters
        filters={filters}
        setFilters={setFilters}
        onReset={resetFilters}
        onApply={applyFilters}
        onChangeSort={onChangeSort}
        sortKey={sortKey}
        sortOrder={sortOrder}
      />
      <ETFTable
        etfs={filteredEtfs}
        sortKey={sortKey}
        sortOrder={sortOrder}
        onSelectETF={onSelectETF} 
      />
    </>
  );
}
