import React, { useMemo } from 'react';
import { ThemeProvider, CssBaseline } from '@mui/material';
import getTheme from './styles/Theme'; // ou chemin correct vers getTheme
import Dashboard from './pages/Dashboard';
import { useTheme } from './context/ThemeContext'; 

export default function App() {
  const { mode } = useTheme(); 
  const theme = useMemo(() => getTheme(mode), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Dashboard />
    </ThemeProvider>
  );
}
