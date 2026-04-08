import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 3000,
    proxy: {
      '/reset': 'http://localhost:7860',
      '/step': 'http://localhost:7860',
      '/state': 'http://localhost:7860',
      '/tasks': 'http://localhost:7860',
      '/health': 'http://localhost:7860',
      '/grade': 'http://localhost:7860',
      '/metrics': 'http://localhost:7860',
      '/leaderboard': 'http://localhost:7860',
      '/sessions': 'http://localhost:7860',
      '/scenarios': 'http://localhost:7860',
      '/protocol': 'http://localhost:7860',
      '/docs': 'http://localhost:7860',
      '/episode': 'http://localhost:7860',
      '/replay': 'http://localhost:7860',
      '/validate': 'http://localhost:7860',
      '/ws': { target: 'ws://localhost:7860', ws: true },
    },
  },
});
