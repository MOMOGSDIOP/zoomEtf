import { createTheme } from '@mui/material/styles';

const getTheme = (mode) =>
  createTheme({
    palette: {
      mode,
      ...(mode === 'dark'
        ? {
            background: {
              default: '#0d1b2a',
              paper: '#1b263b',
            },
            text: {
              primary: '#e0e0e0',
              secondary: '#9fa8da',
            },
            primary: {
              main: '#90caf9',
              contrastText: '#0d1b2a',
            },
            secondary: {
              main: '#f48fb1',
            },
          }
        : {
            background: {
              default: '#fdfdfd',
              paper: '#ffffff',
            },
            text: {
              primary: '#1a1a1a',
              secondary: '#5f6368',
            },
            primary: {
              main: '#1976d2',
              contrastText: '#ffffff',
            },
            secondary: {
              main: '#f50057',
            },
          }),
    },
    typography: {
      fontFamily: "'Inter', 'Roboto', 'Segoe UI', 'Helvetica Neue', sans-serif",
      fontSize: 14,
      h1: { fontSize: '2.5rem', fontWeight: 700 },
      h2: { fontSize: '2rem', fontWeight: 600 },
      h3: { fontSize: '1.75rem', fontWeight: 600 },
      h4: { fontSize: '1.5rem', fontWeight: 500 },
      h5: { fontSize: '1.25rem', fontWeight: 500 },
      h6: { fontSize: '1.1rem', fontWeight: 500 },
      body1: { fontSize: '1rem', lineHeight: 1.6 },
      body2: { fontSize: '0.875rem', lineHeight: 1.5 },
      button: { textTransform: 'none', fontWeight: 600 },
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow:
              mode === 'dark'
                ? '0 4px 20px rgba(0,0,0,0.2)'
                : '0 2px 8px rgba(0,0,0,0.1)',
          },
        },
      },
    },
  });

export default getTheme;
