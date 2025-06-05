import React, { useState, useEffect } from 'react';
import {
  TextField, Button, Box, Typography,
  Paper, InputAdornment, Chip,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Alert
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import etfs from '../data/etfs';

export default function ETFSearcher({ onSelectETF }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState([]);
  const [notFound, setNotFound] = useState(false);
  const [inputError, setInputError] = useState(false);

  // Validation des données ETF
  const isValidETF = (etf) => {
    return etf && 
           typeof etf === 'object' && 
           etf.name && 
           typeof etf.name === 'string';
  };

  const handleSearch = () => {
    setInputError(false);
    setNotFound(false);
    
    const term = searchTerm.trim();
    if (!term) {
      setResults([]);
      setInputError(true);
      return;
    }

    const searchTermLower = term.toLowerCase();
    
    try {
      const matches = etfs.filter(etf => {
        if (!isValidETF(etf)) return false;
        
        return (
          etf.name.toLowerCase().includes(searchTermLower) ||
          (etf.symbol && etf.symbol.toLowerCase().includes(searchTermLower)) ||
          (etf.isin && etf.isin.toLowerCase().includes(searchTermLower))
        );
      });

      setResults(matches);
      setNotFound(matches.length === 0);
    } catch (error) {
      console.error('Erreur lors de la recherche:', error);
      setResults([]);
      setNotFound(true);
    }
  };

  const handleSelect = (etf) => {
    if (!isValidETF(etf)) {
      console.error('ETF invalide sélectionné:', etf);
      return;
    }
    
    if (typeof onSelectETF === 'function') {
      onSelectETF(etf.name);
    } else {
      console.error('onSelectETF n\'est pas une fonction');
    }
  };

  const popularSuggestions = [
    { label: "ETF Tech", filter: "Technologie" },
    { label: "ETF ESG", filter: "ESG" },
    { label: "ETF Europe", filter: "Europe" }
  ];

  // Reset des résultats quand le terme de recherche est vide
  useEffect(() => {
    if (!searchTerm.trim()) {
      setResults([]);
      setNotFound(false);
    }
  }, [searchTerm]);

  return (
    <Paper elevation={3} sx={{ p: 3, position: 'relative' }}>
      <Typography variant="h5" sx={{ mb: 2, color: 'secondary.main' }}>
        🔍 Recherche Avancée
      </Typography>

      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <TextField
          fullWidth
          variant="outlined"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="Rechercher par nom, ISIN ou symbole..."
          error={inputError}
          helperText={inputError ? "Veuillez entrer un terme de recherche" : ""}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon color={inputError ? "error" : "action"} />
              </InputAdornment>
            ),
          }}
        />
        <Button
          variant="contained"
          size="large"
          onClick={handleSearch}
          sx={{ px: 4 }}
          disabled={!searchTerm.trim()}
        >
          Rechercher
        </Button>
      </Box>

      {inputError && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Veuillez entrer un terme de recherche valide
        </Alert>
      )}

      {results.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            {results.length} résultat(s) trouvé(s)
          </Typography>
          <TableContainer component={Paper}>
            <Table sx={{ minWidth: 650 }} aria-label="table des résultats">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 'bold' }}>Nom</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Symbole</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>ISIN</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {results.map((etf) => (
                  <TableRow
                    key={etf.isin || etf.name}
                    hover
                    onClick={() => handleSelect(etf)}
                    sx={{ 
                      cursor: 'pointer',
                      '&:last-child td, &:last-child th': { border: 0 }
                    }}
                  >
                    <TableCell component="th" scope="row">
                      {etf.name}
                    </TableCell>
                    <TableCell>{etf.symbol || '-'}</TableCell>
                    <TableCell>{etf.isin || '-'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      {notFound && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Aucun ETF trouvé pour "{searchTerm}"
        </Alert>
      )}

      <Box sx={{ mt: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold' }}>
          Suggestions rapides :
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          {popularSuggestions.map((suggestion) => (
            <Chip 
              key={suggestion.label}
              icon={<TrendingUpIcon fontSize="small" />}
              label={suggestion.label}
              clickable
              variant="outlined"
              onClick={() => {
                setSearchTerm(suggestion.filter);
                // Déclenche la recherche après un court délai
                setTimeout(handleSearch, 100);
              }}
              sx={{ 
                borderRadius: 1,
                '&:hover': { backgroundColor: 'action.hover' }
              }}
            />
          ))}
        </Box>
      </Box>
    </Paper>
  );
}