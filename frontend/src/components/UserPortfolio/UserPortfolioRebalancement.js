import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../../styles/UserPortfolioFeature.css';
import { Button, Paper, Typography, Box } from '@mui/material';

export default function UserPortfolioRebalancement({ type, onBack, onSelectETF }) {
  const navigate = useNavigate();

  return (
    <div className="feature-page">
      <Box sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        mb: 2
      }}>
        <Typography variant="h4">Rebalancement</Typography>
        <Button variant="outlined" onClick={onBack}>Retour</Button>
      </Box>
      <h4>🔄 Rebalancement automatique</h4>
      <p>Fonctionnalité à venir...</p>
    </div>
  );
}
