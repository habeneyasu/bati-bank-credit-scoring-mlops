import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/predict': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/explain': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    }
  }
})
