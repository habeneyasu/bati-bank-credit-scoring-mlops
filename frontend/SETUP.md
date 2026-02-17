# React Dashboard Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and set your API URL:
```
VITE_API_URL=http://localhost:8001
```

### 3. Start Development Server

```bash
npm run dev
```

The dashboard will open at `http://localhost:3000`

### 4. Make Sure Backend is Running

The React app requires the FastAPI backend to be running on port 8001:

```bash
# In another terminal
cd ..
uvicorn src.api.main:app --reload --port 8001
```

## Features

### 🎨 Modern UI Components
- **Feature Input Form**: Collapsible sections with search functionality
- **Prediction Result**: Visual risk assessment with charts
- **Explanation Panel**: SHAP-based feature importance visualization
- **Scenario Tester**: Interactive feature adjustment tool

### 📊 Visualizations
- Pie charts for risk distribution
- Bar charts for feature importance
- Line charts for scenario history
- Progress bars for risk levels

### 🎯 Key Features
- Real-time predictions
- SHAP explanations
- Scenario testing
- Responsive design
- Professional styling

## Building for Production

```bash
npm run build
```

The production build will be in the `dist` directory.

## Troubleshooting

### API Connection Issues
- Ensure backend is running on the correct port
- Check `.env` file has correct `VITE_API_URL`
- Verify CORS settings in backend

### Build Errors
- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Check Node.js version (requires 18+)

## Development Tips

- Hot reload is enabled - changes reflect immediately
- Check browser console for API errors
- Use React DevTools for debugging
