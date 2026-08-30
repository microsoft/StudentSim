import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const base = process.env.VITE_BASE_PATH ?? './';

export default defineConfig({
  base,
  plugins: [react()],
  server: {
    port: 5274
  },
  preview: {
    port: 5274
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 5000
  }
});
