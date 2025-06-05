import React from 'react';
import {
  Paper, Typography, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow
} from '@mui/material';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import '../styles/ETFList.css';
import etfs from '../data/etfs';

export default function ETFList({ onSelect, onNavigate, loading }) {
  const calculateDailyPerformance = (etf) => (
    ((etf.currentPrice - etf.previousClose) / etf.previousClose) * 100
  );

  const sortedETFs = etfs.filter(etf => etf.currentPrice && etf.previousClose);

  const bestETFs = [...sortedETFs]
    .sort((a, b) => calculateDailyPerformance(b) - calculateDailyPerformance(a))
    .slice(0, 3);

  const worstETFs = [...sortedETFs]
    .sort((a, b) => calculateDailyPerformance(a) - calculateDailyPerformance(b))
    .slice(0, 3);

  const PriceWithArrow = ({ currentPrice, previousClose }) => {
    const perf = calculateDailyPerformance({ currentPrice, previousClose });
    if (perf > 0) return <span>{perf.toFixed(2)} % <ArrowDropUpIcon color="success" /></span>;
    if (perf < 0) return <span>{perf.toFixed(2)} % <ArrowDropDownIcon color="error" /></span>;
    return <span>0.00 %</span>;
  };

  const renderTable = (title, data, viewType) => (
    <Paper className="etf-section" sx={{ padding: 2 }}>
      <Typography
        variant="h6"
        className="etf-title"
        sx={{
          cursor: 'pointer',
          textAlign: 'center',
          marginBottom: 2,
          marginLeft: 4,
          marginRight: 4,
          userSelect: 'none',
        }}
        onClick={() => onNavigate(viewType)} 
      >
        {title}
      </Typography>
      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Nom</TableCell>
              <TableCell align="right">Performance journalière</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.map(etf => (
              <TableRow
                key={etf.name}
                hover
                sx={{ cursor: 'pointer' }}
                onClick={() => onSelect(etf.name)} 
              >
                <TableCell>{etf.name}</TableCell>
                <TableCell align="right">
                  <PriceWithArrow currentPrice={etf.currentPrice} previousClose={etf.previousClose} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );

  return (
    <div className="etf-list-container" style={{ display: 'flex', flexDirection: 'column', gap: '40px' }}>
      {renderTable('Meilleure performance du jour', bestETFs, 'best')}
      {renderTable('Pire performance du jour', worstETFs, 'worst')}
    </div>
  );
}