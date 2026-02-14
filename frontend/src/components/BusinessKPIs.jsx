import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, TrendingDown, Users, CheckCircle, XCircle, Clock, 
  RefreshCw, Calendar, BarChart3, AlertCircle 
} from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const BusinessKPIs = () => {
  const [kpi, setKpi] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [periodType, setPeriodType] = useState('daily');
  const [calculating, setCalculating] = useState(false);

  useEffect(() => {
    loadKPIs();
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
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {/* Total Predictions */}
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-blue-700">Total Predictions</span>
                <BarChart3 className="w-5 h-5 text-blue-600" />
              </div>
              <div className="text-2xl font-bold text-blue-900">
                {formatNumber(kpi.total_predictions)}
              </div>
            </div>

            {/* Unique Customers */}
            <div className="bg-green-50 rounded-lg p-4 border border-green-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-green-700">Unique Customers</span>
                <Users className="w-5 h-5 text-green-600" />
              </div>
              <div className="text-2xl font-bold text-green-900">
                {formatNumber(kpi.unique_customers)}
              </div>
            </div>

            {/* Average Risk Score */}
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-purple-700">Avg Risk Score</span>
                <TrendingUp className="w-5 h-5 text-purple-600" />
              </div>
              <div className="text-2xl font-bold text-purple-900">
                {kpi.avg_risk_score ? kpi.avg_risk_score.toFixed(3) : 'N/A'}
              </div>
            </div>

            {/* Average Latency */}
            <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-orange-700">Avg Latency</span>
                <Clock className="w-5 h-5 text-orange-600" />
              </div>
              <div className="text-2xl font-bold text-orange-900">
                {kpi.avg_latency_ms ? `${kpi.avg_latency_ms.toFixed(0)}ms` : 'N/A'}
              </div>
            </div>
          </div>

          {/* Approval/Rejection Rates */}
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

          {/* Performance Metrics */}
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Performance Metrics</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-sm text-gray-600">P95 Latency</span>
                <div className="text-lg font-semibold text-gray-900">
                  {kpi.p95_latency_ms ? `${kpi.p95_latency_ms.toFixed(0)}ms` : 'N/A'}
                </div>
              </div>
              <div>
                <span className="text-sm text-gray-600">Period</span>
                <div className="text-lg font-semibold text-gray-900 capitalize">
                  {kpi.period_type}
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
