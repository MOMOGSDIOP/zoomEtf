import React, { useState } from 'react';
import ETFFiltersManager from '../ETFFiltersManager';
import ETFIdentity from '../ETFIdentity';

export default function TopLeftQuadrant({ isExpanded }) {
  const [view, setView] = useState('filtre');
  const [selectedETF, setSelectedETF] = useState(null);

  const handleSelectETF = (name) => {
    setSelectedETF(name);
    setView('identity');
  };

  const handleBack = () => {
    setView('filtre');
    setSelectedETF(null);
  };

  const renderView = () => {
    switch (view) {
      case 'filtre':
        return <ETFFiltersManager onSelectETF={handleSelectETF} />;
      case 'identity':
        return <ETFIdentity name={selectedETF} onBack={handleBack} />;
      default:
        return <ETFFiltersManager onSelectETF={handleSelectETF} />;
    }
  };

  return <div className="quadrant top-left">{renderView()}</div>;
}