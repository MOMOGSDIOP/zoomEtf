// semanticETFSearcher.jsx
import React, { useState, useEffect, useCallback } from 'react';
import {
  TextField, Button, Box, Typography,
  Paper, InputAdornment, Chip,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Alert, CircularProgress
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import etfs from '../data/etfs';

import { useNavigate } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
// Fonctions utilitaires externes
import { analyzeUserQuery, extractKeywords } from '../utils/semanticAnalysis';
import { validateETF } from '../utils/etfValidation';

const POPULAR_SUGGESTIONS = [
  { label: "ETF Tech", filter: "Technologie" },
  { label: "ETF ESG", filter: "ESG" },
  { label: "ETF Europe", filter: "Europe" },
  { label: "ETF Dividende", filter: "Dividende" },
  { label: "ETF Obligataire", filter: "Obligataire" }
];

export default function ETFSearcher({ onSelectETF }) {
  const [searchTerm, setSearchTerm] = React.useState('');
  const [results, setResults] = React.useState([]);
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState({ notFound: false, input: false, server: false });
  const [searchMetrics, setSearchMetrics] = useState({ time: 0, keywords: [] });
  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNavigation = (path) => {
    handleMenuClose();
    navigate(path);
  };

  // Recherche avec mémoïsation
  const performSearch = useCallback(async (term) => {
    if (!term.trim()) {
      setResults([]);
      setError(prev => ({ ...prev, notFound: false }));
      return;
    }

    setIsLoading(true);
    const startTime = performance.now();

    try {
      // Analyse sémantique de la requête
      const { keywords, intent } = analyzeUserQuery(term);
      setSearchMetrics(prev => ({ ...prev, keywords }));

      // Filtrage avancé basé sur l'analyse
      const matches = etfs.filter(etf => {
        if (!validateETF(etf)) return false;
        
        const matchesKeywords = keywords.some(keyword => 
          etf.name.toLowerCase().includes(keyword) ||
          (etf.description && etf.description.toLowerCase().includes(keyword)) ||
          (etf.category && etf.category.toLowerCase().includes(keyword))
        );

        const matchesIntent = intent === 'all' || 
          (etf.tags && etf.tags.includes(intent));

        return matchesKeywords && matchesIntent;
      });

      setResults(matches);
      setError({
        notFound: matches.length === 0,
        input: false,
        server: false
      });
    } catch (err) {
      console.error('Erreur de recherche sémantique:', err);
      setError({ notFound: true, input: false, server: true });
      setResults([]);
    } finally {
      setSearchMetrics(prev => ({
        ...prev,
        time: performance.now() - startTime
      }));
      setIsLoading(false);
    }
  }, []);

  // Gestion des recherches avec debounce
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchTerm.trim()) {
        performSearch(searchTerm);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [searchTerm, performSearch]);

  const handleSelect = (etf) => {
    if (!validateETF(etf)) {
      console.error('ETF invalide sélectionné:', etf);
      return;
    }
    
    onSelectETF?.(etf);
  };

  const handleSuggestionClick = (filter) => {
    setSearchTerm(filter);
  };

  return (
    <Paper elevation={3} sx={{ p: 3, position: 'relative' }}>
     <Button onClick={() => handleNavigation('/etfai')}>
       <Typography variant="h5" sx={{ mb: 2, color: 'secondary.main' }}>
        🔍 Recherche Sémantique : ETF.ai
      </Typography>
     </Button>

      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <TextField
          multiline
          minRows={2}
          fullWidth
          variant="outlined"
          value={searchTerm}
          onChange={(e) => {
            setSearchTerm(e.target.value);
            setError(prev => ({ ...prev, input: false }));
          }}
          onKeyDown={(e) => e.key === 'Enter' && performSearch(searchTerm)}
          placeholder="etf technologie avec frais <0.5%, rendement >3% "
          error={error.input}
          helperText={error.input ? "Veuillez entrer une requête valide" : ""}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon color={error.input ? "error" : "action"} />
              </InputAdornment>
            ),
          }}
        />
        <Button
          variant="contained"
          size="large"
          onClick={() => performSearch(searchTerm)}
          sx={{ px: 4 }}
          disabled={!searchTerm.trim() || isLoading}
        >
          {isLoading ? <CircularProgress size={24} /> : "Rechercher"}
        </Button>
      </Box>

      {error.input && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Veuillez saisir une requête de recherche valide
        </Alert>
      )}

      {error.server && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Erreur lors de l'analyse sémantique. Veuillez réessayer.
        </Alert>
      )}

      {isLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
          <CircularProgress />
        </Box>
      )}

      {!isLoading && results.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="subtitle1">
              {results.length} résultat(s) trouvé(s)
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Analyse en {searchMetrics.time.toFixed(0)}ms • 
              Mots-clés: {searchMetrics.keywords.join(', ')}
            </Typography>
          </Box>
          
          <TableContainer component={Paper}>
            <Table sx={{ minWidth: 650 }} aria-label="table des résultats">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 'bold' }}>Nom</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Catégorie</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Frais</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Performance</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {results.map((etf) => (
                  <TableRow
                    key={etf.isin || etf.name}
                    hover
                    onClick={() => handleSelect(etf)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell>
                      <Typography fontWeight="bold">{etf.name}</Typography>
                      <Typography variant="caption">{etf.isin}</Typography>
                    </TableCell>
                    <TableCell>{etf.category || '-'}</TableCell>
                    <TableCell>{etf.fees ? `${etf.fees}%` : '-'}</TableCell>
                    <TableCell>
                      {etf.performance ? `${etf.performance}%` : '-'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      {error.notFound && !isLoading && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Aucun ETF ne correspond à "{searchTerm}". Essayez avec d'autres termes.
        </Alert>
      )}

      <Box sx={{ mt: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold' }}>
          Suggestions populaires :
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          {POPULAR_SUGGESTIONS.map((suggestion) => (
            <Chip 
              key={suggestion.label}
              icon={<TrendingUpIcon fontSize="small" />}
              label={suggestion.label}
              clickable
              variant="outlined"
              onClick={() => handleSuggestionClick(suggestion.filter)}
              sx={{ borderRadius: 1 }}
            />
          ))}
        </Box>
      </Box>
    </Paper>
  );
}