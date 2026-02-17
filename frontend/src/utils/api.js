import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    // Don't set Content-Type for FormData - let browser set it with boundary
    if (config.data instanceof FormData) {
      delete config.headers['Content-Type'];
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Clear token and redirect to login
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user_data');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

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
  async getDataVersions(dataType = null) {
    const params = dataType ? { data_type: dataType } : {};
    const response = await api.get('/api/versions/data', { params });
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

  // Database Service Endpoints
  // Get predictions
  async getPredictions(customerId = null, limit = 100, offset = 0) {
    const params = { limit, offset };
    if (customerId) params.customer_id = customerId;
    const response = await api.get('/api/predictions', { params });
    return response.data;
  },

  // Get prediction by ID
  async getPredictionById(predictionId) {
    const response = await api.get(`/api/predictions/${predictionId}`);
    return response.data;
  },

  // Get customer predictions
  async getCustomerPredictions(customerId, limit = 100) {
    const response = await api.get(`/api/predictions/customer/${customerId}`, {
      params: { limit }
    });
    return response.data;
  },

  // Get business KPIs
  async getKPIs(periodType = null, limit = 100, offset = 0) {
    const params = { limit, offset };
    if (periodType) params.period_type = periodType;
    const response = await api.get('/api/kpis', { params });
    return response.data;
  },

  // Get latest KPIs
  async getLatestKPIs(periodType = 'daily') {
    const response = await api.get('/api/kpis/latest', {
      params: { period_type: periodType }
    });
    return response.data;
  },

  // Calculate KPIs
  async calculateKPIs(periodType = 'daily', hoursBack = 24) {
    const response = await api.post('/api/kpis/calculate', null, {
      params: { period_type: periodType, hours_back: hoursBack }
    });
    return response.data;
  },

  // Get model validation metrics
  async getModelValidationMetrics() {
    const response = await api.get('/api/model/validation-metrics');
    return response.data;
  },

  // User & Role Management Endpoints
  // Get users
  async getUsers(limit = 100, offset = 0, isActive = null) {
    const params = { limit, offset };
    if (isActive !== null) params.is_active = isActive;
    const response = await api.get('/api/users', { params });
    return response.data;
  },

  // Get user by ID
  async getUserById(userId) {
    const response = await api.get(`/api/users/${userId}`);
    return response.data;
  },

  // Get roles
  async getRoles(limit = 100, offset = 0, isActive = null) {
    const params = { limit, offset };
    if (isActive !== null) params.is_active = isActive;
    const response = await api.get('/api/roles', { params });
    return response.data;
  },

  // Get role by ID
  async getRoleById(roleId) {
    const response = await api.get(`/api/roles/${roleId}`);
    return response.data;
  },

  // Get permissions
  async getPermissions() {
    const response = await api.get('/api/permissions');
    return response.data;
  },

  // Authentication endpoints
  async login(username, password) {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);
    const response = await api.post('/api/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  },

  async logout() {
    const response = await api.post('/api/auth/logout');
    return response.data;
  },

  async getCurrentUser() {
    const response = await api.get('/api/auth/me');
    return response.data;
  },

  // Raw Data Management
  async uploadRawData(formData) {
    // Don't set Content-Type header - let axios set it automatically for FormData
    // This ensures the boundary is set correctly
    const response = await api.post('/api/data/upload', formData);
    return response.data;
  },

  async getTransactions(customerId = null, limit = 100, offset = 0, startDate = null, endDate = null) {
    const params = { limit, offset };
    if (customerId) params.customer_id = customerId;
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    const response = await api.get('/api/data/transactions', { params });
    return response.data;
  },

  // Monitoring & Drift Detection
  async detectDrift(featureName = null, modelVersion = null) {
    const params = new URLSearchParams();
    if (featureName) params.append('feature_name', featureName);
    if (modelVersion) params.append('model_version', modelVersion);
    const response = await api.post(`/api/monitoring/drift/detect?${params.toString()}`);
    return response.data;
  },

  async getDriftMetrics(featureName = null, startDate = null, endDate = null, limit = 100) {
    const params = { limit };
    if (featureName) params.feature_name = featureName;
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    const response = await api.get('/api/monitoring/drift/metrics', { params });
    return response.data;
  },

  // Data Quality
  async checkDataQuality() {
    const response = await api.post('/api/monitoring/data-quality/check');
    return response.data;
  },

  // Alerts
  async getAlerts(severity = null, alertType = null, limit = 50) {
    const params = { limit };
    if (severity) params.severity = severity;
    if (alertType) params.alert_type = alertType;
    const response = await api.get('/api/monitoring/alerts', { params });
    return response.data;
  },

  // Lineage
  async getLineage(dataVersionId = null, targetType = null, targetId = null) {
    const params = {};
    if (dataVersionId) params.data_version_id = dataVersionId;
    if (targetType) params.target_type = targetType;
    if (targetId) params.target_id = targetId;
    const response = await api.get('/api/lineage', { params });
    return response.data;
  },

  async getLineageByDataVersion(dataVersionId) {
    const response = await api.get(`/api/lineage/data/${dataVersionId}`);
    return response.data;
  },

  async getLineageByTarget(targetType, targetId) {
    const response = await api.get(`/api/lineage/target/${targetType}/${targetId}`);
    return response.data;
  },

  // Feature Store
  async getCustomerFeatures(customerId) {
    const response = await api.get(`/api/features/${customerId}`);
    return response.data;
  },

  async computeAndStoreFeatures(customerId, transactions, featureVersion = null, dataVersion = null) {
    const response = await api.post(`/api/features/${customerId}`, {
      transactions,
      feature_version: featureVersion,
      data_version: dataVersion
    });
    return response.data;
  },

  async updateCustomerFeatures(customerId, featureVector, options = {}) {
    const response = await api.put(`/api/features/${customerId}`, {
      feature_vector: featureVector,
      ...options
    });
    return response.data;
  },

  async batchGetFeatures(customerIds) {
    const response = await api.post('/api/features/batch', customerIds);
    return response.data;
  },

  async getFeatureStoreStats() {
    const response = await api.get('/api/features/stats');
    return response.data;
  },

  // A/B Testing
  async listExperiments(status = null) {
    const params = status ? { status_filter: status } : {};
    const response = await api.get('/api/experiments', { params });
    return response.data;
  },

  async getExperiment(experimentId) {
    const response = await api.get(`/api/experiments/${experimentId}`);
    return response.data;
  },

  async createExperiment(experimentData) {
    const response = await api.post('/api/experiments', experimentData);
    return response.data;
  },

  async startExperiment(experimentId) {
    const response = await api.post(`/api/experiments/${experimentId}/start`);
    return response.data;
  },

  async stopExperiment(experimentId) {
    const response = await api.post(`/api/experiments/${experimentId}/stop`);
    return response.data;
  },

  async getExperimentResults(experimentId) {
    const response = await api.get(`/api/experiments/${experimentId}/results`);
    return response.data;
  },

  async promoteWinner(experimentId, variantName = null) {
    const response = await api.post(`/api/experiments/${experimentId}/promote`, { variant_name: variantName });
    return response.data;
  },

  // Model Retraining
  async createRetrainingJob(jobData) {
    const response = await api.post('/api/retraining/jobs', jobData);
    return response.data;
  },

  async runRetrainingJob(jobId) {
    const response = await api.post(`/api/retraining/jobs/${jobId}/run`);
    return response.data;
  },

  async listRetrainingJobs(status = null, limit = 20) {
    const params = { limit };
    if (status) params.status_filter = status;
    const response = await api.get('/api/retraining/jobs', { params });
    return response.data;
  },

  async getRetrainingJob(jobId) {
    const response = await api.get(`/api/retraining/jobs/${jobId}`);
    return response.data;
  },

  async createRetrainingSchedule(scheduleData) {
    const response = await api.post('/api/retraining/schedules', scheduleData);
    return response.data;
  },

  async listRetrainingSchedules() {
    const response = await api.get('/api/retraining/schedules');
    return response.data;
  },

  async triggerDriftRetraining(modelName, driftMetadata = {}) {
    const response = await api.post('/api/retraining/trigger/drift', {
      model_name: modelName,
      drift_metadata: driftMetadata
    });
    return response.data;
  },

  // Batch Predictions
  async createBatchPredictionJob(jobData) {
    const response = await api.post('/api/batch-predictions/jobs', jobData);
    return response.data;
  },

  async runBatchPredictionJob(jobId) {
    const response = await api.post(`/api/batch-predictions/jobs/${jobId}/run`);
    return response.data;
  },

  async listBatchPredictionJobs(status = null, limit = 20) {
    const params = { limit };
    if (status) params.status_filter = status;
    const response = await api.get('/api/batch-predictions/jobs', { params });
    return response.data;
  },

  async getBatchPredictionJob(jobId) {
    const response = await api.get(`/api/batch-predictions/jobs/${jobId}`);
    return response.data;
  },

  // Load Testing & Performance Benchmarking
  async runLoadTest(testConfig) {
    const response = await api.post('/api/testing/load-test', testConfig);
    return response.data;
  },

  async estimateCapacity(capacityConfig) {
    const response = await api.post('/api/testing/capacity-planning', capacityConfig);
    return response.data;
  },

  async getBenchmarkResults(limit = 10) {
    const response = await api.get('/api/testing/benchmark', { params: { limit } });
    return response.data;
  },
};

export default creditScoringAPI;
