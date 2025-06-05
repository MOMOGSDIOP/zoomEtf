import React from 'react';
import {
  Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow,
  Paper, Typography
} from '@mui/material';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import '../styles/ETFTable.css';

export default function ETFTable({ etfs = [], sortKey = 'performance', sortOrder = 'desc',onBack, onSelectETF}) {
  // Définition dynamique des colonnes selon sortKey
  const columns = [
    { key: 'name', label: 'Nom', align: 'left' },
    sortKey === 'price'
      ? { key: 'price', label: 'Prix (€)', align: 'right' }
      : { key: 'performance', label: 'Performance (1an)', align: 'right' },
    sortKey === 'price'
      ? { key: 'performance', label: 'Performance (1an)', align: 'right' }
      : { key: 'price', label: 'Prix (€)', align: 'right' },
    { key: 'currentPrice', label: 'Cours actuel', align: 'right' },
    { key: 'issuer', label: 'Émetteur', align: 'left' }
  ];

  // Sous-composant pour afficher cours actuel avec flèche
  const PriceWithArrow = ({ currentPrice, previousClose }) => {
    if (typeof currentPrice !== 'number') return <span>-</span>;

    const priceFixed = currentPrice.toFixed(2);

    let arrow = null;
    if (typeof previousClose === 'number') {
      if (currentPrice > previousClose) {
        arrow = <ArrowDropUpIcon color="success" />;
      } else if (currentPrice < previousClose) {
        arrow = <ArrowDropDownIcon color="error" />;
      }
    }

    return (
      <span>
        {priceFixed} € {arrow}
      </span>
    );
  };

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Typography variant="h5" sx={{ mb: 2 }}>
        Liste complète des ETFs
      </Typography>
      <TableContainer>
        <Table className="etf-table">
          <TableHead>
            <TableRow>
              {columns.map((col) => (
                <TableCell key={col.key} align={col.align}>
                  {col.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {etfs.length > 0 ? (
              etfs.map((etf, i) => (
                <TableRow  key={i} 
                  hover 
                  sx={{ cursor: 'pointer' }}
                  onClick={() => onSelectETF && onSelectETF(etf.name)}>
                  {columns.map((col) => {
                    switch (col.key) {
                      case 'name':
                        return <TableCell key={col.key}>{etf.name}</TableCell>;

                      case 'price':
                        return (
                          <TableCell key={col.key} align={col.align}>
                            {typeof etf.price === 'number' ? etf.price.toFixed(2) + ' €' : '-'}
                          </TableCell>
                        );

                      case 'performance':
                        return (
                          <TableCell
                            key={col.key}
                            align={col.align}
                            sx={{ color: etf.performance >= 0 ? 'success.main' : 'error.main' }}
                          >
                            {etf.performance > 0 ? '+' : ''}
                            {typeof etf.performance === 'number' ? etf.performance.toFixed(2) : '-'}%
                          </TableCell>
                        );

                      case 'currentPrice':
                        return (
                          <TableCell key={col.key} align={col.align}>
                            <PriceWithArrow
                              currentPrice={etf.currentPrice}
                              previousClose={etf.previousClose}
                            />
                          </TableCell>
                        );

                      case 'issuer':
                        return <TableCell key={col.key}>{etf.issuer || '-'}</TableCell>;

                      default:
                        return <TableCell key={col.key}>-</TableCell>;
                    }
                  })}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} align="center">
                  Aucun ETF trouvé.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
}
