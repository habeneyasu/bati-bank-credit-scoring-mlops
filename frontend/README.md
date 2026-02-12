# Credit Scoring Dashboard - React Frontend

A modern, interactive React.js dashboard for credit risk assessment. Built for loan officers and credit analysts to explore risk profiles, test scenarios, and understand predictions without writing code.

## Features

- 🎨 **Modern UI**: Beautiful, professional design with Tailwind CSS
- 📊 **Interactive Visualizations**: Charts and graphs using Recharts
- 🧠 **SHAP Explanations**: Visual feature importance analysis
- 🔄 **Scenario Testing**: Adjust features and see real-time impact
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- ⚡ **Fast Performance**: Built with Vite for optimal performance

## Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8001`

## Installation

```bash
cd frontend
npm install
```

## Development

Start the development server:

```bash
npm run dev
```

The dashboard will be available at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Environment Variables

Create a `.env` file in the `frontend` directory:

```env
VITE_API_URL=http://localhost:8001
```

## Project Structure

```
frontend/
├── src/
│   ├── components/      # React components
│   │   ├── FeatureInputForm.jsx
│   │   ├── PredictionResult.jsx
│   │   ├── ExplanationPanel.jsx
│   │   └── ScenarioTester.jsx
│   ├── pages/          # Page components
│   │   └── Dashboard.jsx
│   ├── utils/          # Utilities
│   │   └── api.js      # API client
│   ├── styles/         # CSS files
│   │   └── index.css
│   ├── App.jsx         # Main app component
│   └── main.jsx        # Entry point
├── public/             # Static assets
├── package.json
└── vite.config.js
```

## Usage

1. **Enter Customer Features**: Use the feature input form to enter customer data
2. **Get Prediction**: Click "Predict" to get risk assessment
3. **View Explanation**: Click "Explain" to see SHAP-based feature importance
4. **Test Scenarios**: Adjust feature values and see how they affect risk

## Technologies

- **React 18**: UI library
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **Recharts**: Data visualization
- **Axios**: HTTP client
- **Lucide React**: Icons

## License

MIT
