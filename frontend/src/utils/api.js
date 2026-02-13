import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const creditScoringAPI = {
  // Get feature names
  async getFeatureNames() {
    const response = await api.get('/api/feature-names');
    return response.data;
  },

  // Get prediction
  async predict(features, includeExplanation = false) {
    const response = await api.post('/predict', {
      features,
      include_explanation: includeExplanation,
    });
    return response.data;
  },

  // Get explanation
  async explain(features, includePlot = false) {
    const response = await api.post('/explain', {
      features,
    }, {
      params: { include_plot: includePlot },
    });
    return response.data;
  },

  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Get fairness analysis
  async getFairnessAnalysis() {
    const response = await api.get('/api/fairness');
    return response.data;
  },

  // Get all versions
  async getVersions() {
    const response = await api.get('/api/versions');
    return response.data;
  },

  // Get model versions
  async getModelVersions() {
    const response = await api.get('/api/versions/model');
    return response.data;
  },

  // Get data versions
  async getDataVersions() {
    const response = await api.get('/api/versions/data');
    return response.data;
  },

  // Get current versions
  async getCurrentVersions() {
    const response = await api.get('/api/versions/current');
    return response.data;
  },

  // Get performance metrics
  async getPerformanceMetrics() {
    const response = await api.get('/api/performance', {
      timeout: 5000 // 5 second timeout
    });
    return response.data;
  },
};

export default creditScoringAPI;
