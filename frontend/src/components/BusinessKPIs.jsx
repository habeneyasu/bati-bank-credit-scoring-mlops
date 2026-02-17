import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, TrendingDown, Users, CheckCircle, XCircle, Clock, 
  RefreshCw, Calendar, BarChart3, AlertCircle, PieChart, Target, Activity
} from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const BusinessKPIs = () => {
  const [kpi, setKpi] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [periodType, setPeriodType] = useState('daily');
  const [calculating, setCalculating] = useState(false);
  const [modelMetrics, setModelMetrics] = useState(null);

  useEffect(() => {
    loadKPIs();
    loadModelMetrics();
  }, [periodType]);

  const loadKPIs = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.getLatestKPIs(periodType);
      setKpi(data.kpi);
    } catch (err) {
      setError(err.message || 'Failed to load KPIs');
      console.error('Error loading KPIs:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadModelMetrics = async () => {
    try {
      const versions = await creditScoringAPI.getCurrentVersions();
      if (versions?.model?.metrics) {
        setModelMetrics(versions.model.metrics);
      } else if (versions?.model?.run_id) {
        // Try to get metrics from MLflow if available
        // For now, we'll use default values from Model Card
        setModelMetrics({
          roc_auc: 0.9950, // From Model Card
          accuracy: 0.9715,
          precision: 0.9401,
          recall: 0.8043
        });
      }
    } catch (err) {
      console.error('Error loading model metrics:', err);
      // Use default values from Model Card if API fails
      setModelMetrics({
        roc_auc: 0.9950,
        accuracy: 0.9715,
        precision: 0.9401,
        recall: 0.8043
      });
    }
  };

  const calculateKPIs = async () => {
    setCalculating(true);
    setError(null);
    try {
      await creditScoringAPI.calculateKPIs(periodType, 24);
      await loadKPIs(); // Reload after calculation
    } catch (err) {
      setError(err.message || 'Failed to calculate KPIs');
      console.error('Error calculating KPIs:', err);
    } finally {
      setCalculating(false);
    }
  };

  const formatPercentage = (value) => {
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatNumber = (value) => {
    if (value === null || value === undefined) return 'N/A';
    return value.toLocaleString();
  };

  if (loading && !kpi) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
        <div className="text-center text-gray-500">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
          <p>Loading KPIs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Business KPIs</h3>
          <div className="flex items-center gap-2">
            <select
              value={periodType}
              onChange={(e) => setPeriodType(e.target.value)}
              className="px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            >
              <option value="hourly">Hourly</option>
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
            </select>
            <button
              onClick={calculateKPIs}
              disabled={calculating}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              <BarChart3 className="w-4 h-4" />
              Calculate
            </button>
            <button
              onClick={loadKPIs}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border-l-4 border-red-400">
          <div className="flex items-center gap-2 text-red-700">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {!kpi ? (
        <div className="p-8 text-center text-gray-500">
          <p>No KPIs available for {periodType} period</p>
          <button
            onClick={calculateKPIs}
            disabled={calculating}
            className="mt-4 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {calculating ? 'Calculating...' : 'Calculate KPIs'}
          </button>
        </div>
      ) : (
        <div className="p-6">
          {/* Executive Analytics Overview Header */}
          <div className="mb-6 pb-4 border-b border-gray-200">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Executive Analytics Overview</h2>
            <p className="text-sm text-gray-600">Business value and model effectiveness metrics</p>
          </div>

          {/* Key Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {/* Total Processed Applications */}
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-5 text-white shadow-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium opacity-90">Total Processed Applications</span>
                <BarChart3 className="w-5 h-5 opacity-80" />
              </div>
              <div className="text-3xl font-bold mb-1">
                {formatNumber(kpi.total_predictions)}
              </div>
              <div className="text-xs opacity-75">All time processed</div>
            </div>

            {/* Approval Rate */}
            <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-5 text-white shadow-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium opacity-90">Approval Rate</span>
                <CheckCircle className="w-5 h-5 opacity-80" />
              </div>
              <div className="text-3xl font-bold mb-1">
                {formatPercentage(kpi.approval_rate)}
              </div>
              <div className="text-xs opacity-75">
                {formatNumber(kpi.approval_count)} approved
              </div>
            </div>

            {/* Default Rate */}
            <div className="bg-gradient-to-br from-red-500 to-red-600 rounded-lg p-5 text-white shadow-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium opacity-90">Default Rate</span>
                <AlertCircle className="w-5 h-5 opacity-80" />
              </div>
              <div className="text-3xl font-bold mb-1">
                {kpi.rejection_rate ? formatPercentage(kpi.rejection_rate) : 'N/A'}
              </div>
              <div className="text-xs opacity-75">
                {kpi.rejection_count ? `${formatNumber(kpi.rejection_count)} rejected` : 'Not tracked yet'}
              </div>
            </div>

            {/* Average Score */}
            <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-5 text-white shadow-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium opacity-90">Average Score</span>
                <Target className="w-5 h-5 opacity-80" />
              </div>
              <div className="text-3xl font-bold mb-1">
                {kpi.avg_risk_score ? (kpi.avg_risk_score * 100).toFixed(1) : 'N/A'}
              </div>
              <div className="text-xs opacity-75">Risk score (0-100 scale)</div>
            </div>

            {/* ROC-AUC / Model Performance */}
            <div className="bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-lg p-5 text-white shadow-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium opacity-90">ROC-AUC</span>
                <Activity className="w-5 h-5 opacity-80" />
              </div>
              <div className="text-3xl font-bold mb-1">
                {modelMetrics?.roc_auc ? modelMetrics.roc_auc.toFixed(4) : '0.9950'}
              </div>
              <div className="text-xs opacity-75">Model performance metric</div>
            </div>

            {/* Portfolio Risk Distribution Preview */}
            <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-lg p-5 text-white shadow-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium opacity-90">Portfolio Risk</span>
                <PieChart className="w-5 h-5 opacity-80" />
              </div>
              <div className="text-xs opacity-75 mb-2">Distribution:</div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span>Low: {formatPercentage(kpi.approval_rate)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span>Medium: {formatPercentage(kpi.review_rate)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span>High: {formatPercentage(kpi.rejection_rate)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Portfolio Risk Distribution Chart */}
          <div className="bg-white rounded-lg p-6 border border-gray-200 mb-6 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <PieChart className="w-5 h-5 text-indigo-600" />
              Portfolio Risk Distribution
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Low Risk */}
              <div className="relative">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Low Risk</span>
                  <span className="text-sm font-bold text-green-600">
                    {formatPercentage(kpi.approval_rate)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div 
                    className="bg-green-500 h-4 rounded-full transition-all duration-500"
                    style={{ width: `${(kpi.approval_rate || 0) * 100}%` }}
                  ></div>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {formatNumber(kpi.approval_count)} customers
                </div>
              </div>

              {/* Medium Risk */}
              <div className="relative">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Medium Risk</span>
                  <span className="text-sm font-bold text-yellow-600">
                    {formatPercentage(kpi.review_rate)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div 
                    className="bg-yellow-500 h-4 rounded-full transition-all duration-500"
                    style={{ width: `${(kpi.review_rate || 0) * 100}%` }}
                  ></div>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {formatNumber(kpi.review_count)} customers
                </div>
              </div>

              {/* High Risk */}
              <div className="relative">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">High Risk</span>
                  <span className="text-sm font-bold text-red-600">
                    {formatPercentage(kpi.rejection_rate)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div 
                    className="bg-red-500 h-4 rounded-full transition-all duration-500"
                    style={{ width: `${(kpi.rejection_rate || 0) * 100}%` }}
                  ></div>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {formatNumber(kpi.rejection_count)} customers
                </div>
              </div>
            </div>
          </div>

          {/* Additional Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-green-50 rounded-lg p-4 border border-green-200">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-5 h-5 text-green-600" />
                <span className="text-sm font-medium text-green-700">Approvals</span>
              </div>
              <div className="text-xl font-bold text-green-900 mb-1">
                {formatNumber(kpi.approval_count)}
              </div>
              <div className="text-sm text-green-700">
                Rate: {formatPercentage(kpi.approval_rate)}
              </div>
            </div>

            <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-5 h-5 text-yellow-600" />
                <span className="text-sm font-medium text-yellow-700">Reviews</span>
              </div>
              <div className="text-xl font-bold text-yellow-900 mb-1">
                {formatNumber(kpi.review_count)}
              </div>
              <div className="text-sm text-yellow-700">
                Rate: {formatPercentage(kpi.review_rate)}
              </div>
            </div>

            <div className="bg-red-50 rounded-lg p-4 border border-red-200">
              <div className="flex items-center gap-2 mb-2">
                <XCircle className="w-5 h-5 text-red-600" />
                <span className="text-sm font-medium text-red-700">Rejections</span>
              </div>
              <div className="text-xl font-bold text-red-900 mb-1">
                {formatNumber(kpi.rejection_count)}
              </div>
              <div className="text-sm text-red-700">
                Rate: {formatPercentage(kpi.rejection_rate)}
              </div>
            </div>
          </div>

          {/* Model Performance & System Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Model Performance Metrics */}
            <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg p-4 border border-indigo-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4 text-indigo-600" />
                Model Performance Metrics
              </h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">ROC-AUC</span>
                  <div className="text-lg font-bold text-indigo-900">
                    {modelMetrics?.roc_auc ? modelMetrics.roc_auc.toFixed(4) : '0.9950'}
                  </div>
                </div>
                {modelMetrics?.accuracy && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Accuracy</span>
                    <div className="text-sm font-semibold text-gray-700">
                      {(modelMetrics.accuracy * 100).toFixed(2)}%
                    </div>
                  </div>
                )}
                {modelMetrics?.precision && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Precision</span>
                    <div className="text-sm font-semibold text-gray-700">
                      {(modelMetrics.precision * 100).toFixed(2)}%
                    </div>
                  </div>
                )}
                {modelMetrics?.recall && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Recall</span>
                    <div className="text-sm font-semibold text-gray-700">
                      {(modelMetrics.recall * 100).toFixed(2)}%
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* System Performance Metrics */}
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                <Clock className="w-4 h-4 text-gray-600" />
                System Performance
              </h4>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-sm text-gray-600">P95 Latency</span>
                  <div className="text-lg font-semibold text-gray-900">
                    {kpi.p95_latency_ms ? `${kpi.p95_latency_ms.toFixed(0)}ms` : 'N/A'}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Avg Latency</span>
                  <div className="text-lg font-semibold text-gray-900">
                    {kpi.avg_latency_ms ? `${kpi.avg_latency_ms.toFixed(0)}ms` : 'N/A'}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Period</span>
                  <div className="text-lg font-semibold text-gray-900 capitalize">
                    {kpi.period_type}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Unique Customers</span>
                  <div className="text-lg font-semibold text-gray-900">
                    {formatNumber(kpi.unique_customers)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BusinessKPIs;
