import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Support multiple mount points for OJS integration
const mountElement = document.getElementById('skz-dashboard-root') || document.getElementById('root');

if (mountElement) {
  createRoot(mountElement).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
