import React from 'react';
import { Button, Paper, Typography } from '@mui/material';
import etfs from '../data/etfs';
import '../styles/ETFIdentity.css';

export default function ETFIdentity({ name, onBack }) {
  const etf = etfs.find(e => e.name === name);

  if (!etf) return <Typography>ETF introuvable</Typography>;

  return (
    <Paper className="etf-identity">
      <Typography variant="h5" gutterBottom>{etf.name}</Typography>
      <Typography><strong>Prix actuel :</strong> {etf.currentPrice} €</Typography>
      <Typography><strong>Clôture précédente :</strong> {etf.previousClose} €</Typography>
      <Typography><strong>Performance annuelle :</strong> {etf.performance}%</Typography>
      <Typography><strong>Émetteur :</strong> {etf.issuer}</Typography>
      <Typography><strong>Région :</strong> {etf.region}</Typography>
      <Typography><strong>Type :</strong> {etf.type}</Typography>
      <Typography><strong>Réplication :</strong> {etf.replication}</Typography>
      <Typography><strong>Secteur :</strong> {etf.sector}</Typography>
      <Typography><strong>Disponibilité :</strong> {etf.availability}</Typography>

      <Button variant="outlined" sx={{ mt: 2 }} onClick={onBack}>
        Retour
      </Button>
    </Paper>
  );
}
