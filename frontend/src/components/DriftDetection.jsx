import React, { useState, useEffect } from 'react';
import { AlertTriangle, TrendingUp, TrendingDown, Activity, RefreshCw, BarChart3, CheckCircle, XCircle } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const DriftDetection = () => {
  const [driftMetrics, setDriftMetrics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [detecting, setDetecting] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [featureFilter, setFeatureFilter] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  useEffect(() => {
    loadDriftMetrics();
  }, [featureFilter, startDate, endDate]);

  const loadDriftMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.getDriftMetrics(featureFilter || null, startDate || null, endDate || null);
      setDriftMetrics(data.metrics || []);
    } catch (err) {
      setError(err.message || 'Failed to load drift metrics');
      console.error('Error loading drift metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDetectDrift = async () => {
    setDetecting(true);
    setError(null);
    setDetectionResult(null);
    try {
      const result = await creditScoringAPI.detectDrift();
      setDetectionResult(result);
      // Reload metrics after detection
      setTimeout(() => loadDriftMetrics(), 1000);
    } catch (err) {
      setError(err.message || 'Failed to detect drift');
      console.error('Error detecting drift:', err);
    } finally {
      setDetecting(false);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'major':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'minor':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'none':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getPSIColor = (psi) => {
    if (psi >= 0.25) return 'text-red-600';
    if (psi >= 0.2) return 'text-yellow-600';
    return 'text-green-600';
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Drift Detection</h2>
          <p className="text-sm text-gray-600 mt-1">Monitor data and prediction distribution changes</p>
        </div>
        <button
          onClick={handleDetectDrift}
          disabled={detecting}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Activity className={`w-4 h-4 ${detecting ? 'animate-spin' : ''}`} />
          {detecting ? 'Detecting...' : 'Run Drift Detection'}
        </button>
      </div>

      {/* Detection Result */}
      {detectionResult && (
        <div className={`p-4 rounded-lg border ${
          detectionResult.drift_detected 
            ? 'bg-yellow-50 border-yellow-200' 
            : 'bg-green-50 border-green-200'
        }`}>
          <div className="flex items-start gap-3">
            {detectionResult.drift_detected ? (
              <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
            ) : (
              <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
            )}
            <div className="flex-1">
              <h3 className="font-semibold text-gray-900">
                {detectionResult.drift_detected ? 'Drift Detected' : 'No Drift Detected'}
              </h3>
              <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">PSI:</span>
                  <span className={`ml-2 font-semibold ${getPSIColor(detectionResult.psi)}`}>
                    {detectionResult.psi?.toFixed(4) || 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">KS Statistic:</span>
                  <span className="ml-2 font-semibold text-gray-900">
                    {detectionResult.ks_statistic?.toFixed(4) || 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">KS P-value:</span>
                  <span className="ml-2 font-semibold text-gray-900">
                    {detectionResult.ks_p_value?.toFixed(4) || 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Severity:</span>
                  <span className={`ml-2 px-2 py-0.5 rounded text-xs font-medium ${getSeverityColor(detectionResult.drift_severity)}`}>
                    {detectionResult.drift_severity || 'none'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <input
            type="text"
            placeholder="Filter by feature name"
            value={featureFilter}
            onChange={(e) => setFeatureFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          />
          <input
            type="date"
            placeholder="Start Date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          />
          <div className="flex gap-2">
            <input
              type="date"
              placeholder="End Date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
            />
            <button
              onClick={loadDriftMetrics}
              disabled={loading}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 border-l-4 border-red-400">
          <div className="flex items-center gap-2 text-red-700">
            <XCircle className="w-5 h-5" />
            <span className="text-sm">{error}</span>
          </div>
        </div>
      )}

      {/* Metrics Table */}
      {loading && driftMetrics.length === 0 ? (
        <div className="p-8 text-center text-gray-500">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
          <p>Loading drift metrics...</p>
        </div>
      ) : driftMetrics.length === 0 ? (
        <div className="p-8 text-center text-gray-500 bg-white rounded-lg border border-gray-200">
          <BarChart3 className="w-12 h-12 mx-auto mb-2 text-gray-400" />
          <p>No drift metrics found</p>
          <p className="text-sm mt-1">Run drift detection to generate metrics</p>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Time
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Feature
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    PSI
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    KS Statistic
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Severity
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Model Version
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {driftMetrics.map((metric) => (
                  <tr key={metric.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                      {formatDate(metric.time)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <code className="text-xs text-gray-700 bg-gray-100 px-2 py-1 rounded">
                        {metric.feature_name}
                      </code>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`text-sm font-semibold ${getPSIColor(metric.psi)}`}>
                        {metric.psi?.toFixed(4) || 'N/A'}
                      </span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                      {metric.ks_statistic?.toFixed(4) || 'N/A'}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      {metric.is_drifted ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                          <AlertTriangle className="w-3 h-3" />
                          Drifted
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          <CheckCircle className="w-3 h-3" />
                          Stable
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getSeverityColor(metric.drift_severity)}`}>
                        {metric.drift_severity || 'none'}
                      </span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                      {metric.model_version || 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Summary Stats */}
      {driftMetrics.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Metrics</p>
                <p className="text-2xl font-bold text-gray-900">{driftMetrics.length}</p>
              </div>
              <BarChart3 className="w-8 h-8 text-blue-500" />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Drifted Features</p>
                <p className="text-2xl font-bold text-red-600">
                  {driftMetrics.filter(m => m.is_drifted).length}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-red-500" />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Major Drift</p>
                <p className="text-2xl font-bold text-red-600">
                  {driftMetrics.filter(m => m.drift_severity === 'major').length}
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Avg PSI</p>
                <p className="text-2xl font-bold text-gray-900">
                  {driftMetrics.length > 0
                    ? (driftMetrics.reduce((sum, m) => sum + (m.psi || 0), 0) / driftMetrics.length).toFixed(4)
                    : '0.0000'}
                </p>
              </div>
              <Activity className="w-8 h-8 text-blue-500" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DriftDetection;
