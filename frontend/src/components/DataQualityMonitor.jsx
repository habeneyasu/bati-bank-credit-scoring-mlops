import React, { useState, useEffect } from 'react';
import { CheckCircle, XCircle, AlertTriangle, RefreshCw, Database, TrendingUp, FileCheck, Clock } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const DataQualityMonitor = () => {
  const [qualityReport, setQualityReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadQualityReport();
  }, []);

  const loadQualityReport = async () => {
    setLoading(true);
    setError(null);
    try {
      const report = await creditScoringAPI.checkDataQuality();
      setQualityReport(report);
    } catch (err) {
      setError(err.message || 'Failed to load data quality report');
      console.error('Error loading quality report:', err);
    } finally {
      setLoading(false);
    }
  };

  const getQualityScoreColor = (score) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getQualityScoreBg = (score) => {
    if (score >= 90) return 'bg-green-100 border-green-200';
    if (score >= 70) return 'bg-yellow-100 border-yellow-200';
    return 'bg-red-100 border-red-200';
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  if (loading && !qualityReport) {
    return (
      <div className="p-8 text-center text-gray-500">
        <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
        <p>Loading data quality report...</p>
      </div>
    );
  }

  if (error && !qualityReport) {
    return (
      <div className="p-4 bg-red-50 border-l-4 border-red-400">
        <div className="flex items-center gap-2 text-red-700">
          <XCircle className="w-5 h-5" />
          <span className="text-sm">{error}</span>
        </div>
      </div>
    );
  }

  if (!qualityReport) {
    return (
      <div className="p-8 text-center text-gray-500 bg-white rounded-lg border border-gray-200">
        <Database className="w-12 h-12 mx-auto mb-2 text-gray-400" />
        <p>No quality report available</p>
      </div>
    );
  }

  const summary = qualityReport.summary || {};
  const schemaValidation = qualityReport.schema_validation || {};
  const missingValues = qualityReport.missing_values || {};
  const completeness = qualityReport.completeness || {};
  const freshness = qualityReport.freshness || {};

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Data Quality Monitor</h2>
          <p className="text-sm text-gray-600 mt-1">Monitor data quality metrics and validation</p>
        </div>
        <button
          onClick={loadQualityReport}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Quality Score */}
      <div className={`p-6 rounded-lg border-2 ${getQualityScoreBg(qualityReport.quality_score || 0)}`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600">Overall Quality Score</p>
            <p className={`text-4xl font-bold mt-2 ${getQualityScoreColor(qualityReport.quality_score || 0)}`}>
              {qualityReport.quality_score?.toFixed(1) || 0}
              <span className="text-2xl">/100</span>
            </p>
          </div>
          <div className="text-right">
            {qualityReport.quality_score >= 90 ? (
              <CheckCircle className="w-16 h-16 text-green-600" />
            ) : qualityReport.quality_score >= 70 ? (
              <AlertTriangle className="w-16 h-16 text-yellow-600" />
            ) : (
              <XCircle className="w-16 h-16 text-red-600" />
            )}
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Records</p>
              <p className="text-2xl font-bold text-gray-900">
                {qualityReport.total_records || summary.total_records || 0}
              </p>
            </div>
            <Database className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        <div className={`bg-white rounded-lg shadow-sm border p-4 ${
          schemaValidation.valid ? 'border-green-200' : 'border-red-200'
        }`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Schema Valid</p>
              <p className="text-2xl font-bold">
                {schemaValidation.valid ? (
                  <span className="text-green-600">Yes</span>
                ) : (
                  <span className="text-red-600">No</span>
                )}
              </p>
            </div>
            {schemaValidation.valid ? (
              <CheckCircle className="w-8 h-8 text-green-500" />
            ) : (
              <XCircle className="w-8 h-8 text-red-500" />
            )}
          </div>
        </div>
        <div className={`bg-white rounded-lg shadow-sm border p-4 ${
          completeness.completeness_score >= 90 ? 'border-green-200' : 'border-yellow-200'
        }`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Completeness</p>
              <p className="text-2xl font-bold text-gray-900">
                {completeness.completeness_score?.toFixed(1) || 0}%
              </p>
            </div>
            <FileCheck className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        <div className={`bg-white rounded-lg shadow-sm border p-4 ${
          freshness.fresh ? 'border-green-200' : 'border-yellow-200'
        }`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Data Freshness</p>
              <p className="text-2xl font-bold text-gray-900">
                {freshness.fresh ? 'Fresh' : 'Stale'}
              </p>
            </div>
            <Clock className="w-8 h-8 text-blue-500" />
          </div>
        </div>
      </div>

      {/* Schema Validation Details */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Schema Validation</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Validation Status</span>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              schemaValidation.valid
                ? 'bg-green-100 text-green-800'
                : 'bg-red-100 text-red-800'
            }`}>
              {schemaValidation.valid ? 'Valid' : 'Invalid'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Total Records</span>
            <span className="text-sm font-medium text-gray-900">
              {schemaValidation.total_records || 0}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Errors</span>
            <span className="text-sm font-medium text-red-600">
              {schemaValidation.error_count || 0}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Warnings</span>
            <span className="text-sm font-medium text-yellow-600">
              {schemaValidation.warning_count || 0}
            </span>
          </div>
          {schemaValidation.errors && schemaValidation.errors.length > 0 && (
            <details className="mt-3">
              <summary className="text-sm text-gray-600 cursor-pointer hover:text-gray-900">
                View Errors ({schemaValidation.errors.length})
              </summary>
              <div className="mt-2 p-3 bg-red-50 rounded text-xs max-h-40 overflow-y-auto">
                {schemaValidation.errors.slice(0, 10).map((err, idx) => (
                  <div key={idx} className="text-red-700">{err}</div>
                ))}
                {schemaValidation.errors.length > 10 && (
                  <div className="text-red-600 mt-2">
                    ... and {schemaValidation.errors.length - 10} more errors
                  </div>
                )}
              </div>
            </details>
          )}
        </div>
      </div>

      {/* Missing Values */}
      {missingValues.fields_above_threshold && missingValues.fields_above_threshold.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-yellow-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Missing Values</h3>
          <div className="space-y-2">
            <p className="text-sm text-gray-600">
              Fields with missing values above threshold ({missingValues.threshold}%):
            </p>
            <div className="flex flex-wrap gap-2">
              {missingValues.fields_above_threshold.map((field, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs font-medium"
                >
                  {field} ({missingValues.missing_percentages?.[field]?.toFixed(1) || 0}%)
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Completeness Details */}
      {completeness.field_completeness && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Field Completeness</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {Object.entries(completeness.field_completeness).map(([field, stats]) => (
              <div key={field} className="flex items-center justify-between py-2 border-b border-gray-100">
                <span className="text-sm text-gray-700 font-mono">{field}</span>
                <div className="flex items-center gap-4">
                  <span className="text-xs text-gray-500">
                    {stats.completeness_percentage?.toFixed(1) || 0}%
                  </span>
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        stats.completeness_percentage >= 90
                          ? 'bg-green-500'
                          : stats.completeness_percentage >= 70
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${stats.completeness_percentage || 0}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Freshness Details */}
      {freshness.hours_since_update !== undefined && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Freshness</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Status</span>
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                freshness.fresh
                  ? 'bg-green-100 text-green-800'
                  : 'bg-yellow-100 text-yellow-800'
              }`}>
                {freshness.fresh ? 'Fresh' : 'Stale'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Hours Since Update</span>
              <span className="text-sm font-medium text-gray-900">
                {freshness.hours_since_update?.toFixed(1) || 'N/A'}
              </span>
            </div>
            {freshness.latest_timestamp && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Latest Timestamp</span>
                <span className="text-sm text-gray-700">
                  {formatDate(freshness.latest_timestamp)}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Timestamp */}
      <div className="text-xs text-gray-500 text-center">
        Report generated: {formatDate(qualityReport.timestamp)}
      </div>
    </div>
  );
};

export default DataQualityMonitor;
