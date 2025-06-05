import React from 'react';
import { Button } from '@mui/material';
import { useTheme } from '../context/ThemeContext';
import '../styles/GridLayout.css';
import ErrorBoundary from '../components/ErrorBoundary';
import TopLeftQuadrant from '../components/quadrants/TopLeftQuadrant';
import TopRightQuadrant from '../components/quadrants/TopRightQuadrant';
import BottomLeftQuadrant from '../components/quadrants/BottomLeftQuadrant';
import BottomRightQuadrant from '../components/quadrants/BottomRightQuadrant';

export default function Dashboard() {
  const { toggleTheme, mode } = useTheme();

  return (
    <ErrorBoundary>
      <div className="dashboard-grid">
        <div className="quadrant top-left">
          <TopLeftQuadrant />
        </div>
        <div className="quadrant top-right">
          <TopRightQuadrant />
        </div>
        <div className="quadrant bottom-left">
          <BottomLeftQuadrant />
        </div>
        <div className="quadrant bottom-right">
          <BottomRightQuadrant />
        </div>

        <Button 
          variant="outlined" 
          onClick={toggleTheme}
          style={{ 
            position: 'fixed',
            bottom: 16,
            left: 16,
            zIndex: 1000
          }}
        >
          {mode === 'dark' ? 'Light Mode' : 'Dark Mode'}
        </Button>
      </div>
    </ErrorBoundary>
  );
}