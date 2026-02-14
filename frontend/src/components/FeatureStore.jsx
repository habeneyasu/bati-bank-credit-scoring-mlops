import React, { useState, useEffect } from 'react';
import { 
  Database, RefreshCw, TrendingUp, Clock, CheckCircle, 
  AlertCircle, Package, Activity, Zap, BarChart3 
} from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const FeatureStore = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.getFeatureStoreStats();
      setStats(data);
    } catch (err) {
      setError(err.message || 'Failed to load feature store statistics');
      console.error('Error loading feature store stats:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  if (loading && !stats) {
    return (
      <div className="p-8 text-center text-gray-500">
        <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
        <p>Loading feature store statistics...</p>
      </div>
    );
  }

  if (error && !stats) {
    return (
      <div className="p-4 bg-red-50 border-l-4 border-red-400">
        <div className="flex items-center gap-2 text-red-700">
          <AlertCircle className="w-5 h-5" />
          <span className="text-sm">{error}</span>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="p-8 text-center text-gray-500 bg-white rounded-lg border border-gray-200">
        <Database className="w-12 h-12 mx-auto mb-2 text-gray-400" />
        <p>No feature store statistics available</p>
      </div>
    );
  }

  const cacheCoverage = stats.cache_coverage || 0;
  const coverageColor = cacheCoverage >= 50 ? 'text-green-600' : cacheCoverage >= 25 ? 'text-yellow-600' : 'text-red-600';
  const coverageBg = cacheCoverage >= 50 ? 'bg-green-100 border-green-200' : cacheCoverage >= 25 ? 'bg-yellow-100 border-yellow-200' : 'bg-red-100 border-red-200';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Feature Store</h2>
          <p className="text-sm text-gray-600 mt-1">Pre-computed features for fast online serving</p>
        </div>
        <button
          onClick={loadStats}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">Total Features</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {stats.total_features?.toLocaleString() || 0}
              </p>
            </div>
            <div className="p-3 bg-blue-100 rounded-lg">
              <Database className="w-6 h-6 text-blue-600" />
            </div>
          </div>
          <p className="text-xs text-gray-500">Customers with cached features</p>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">Updated (24h)</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {stats.recent_features_24h?.toLocaleString() || 0}
              </p>
            </div>
            <div className="p-3 bg-green-100 rounded-lg">
              <Clock className="w-6 h-6 text-green-600" />
            </div>
          </div>
          <p className="text-xs text-gray-500">Features refreshed today</p>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">Updated (7d)</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {stats.recent_features_7d?.toLocaleString() || 0}
              </p>
            </div>
            <div className="p-3 bg-purple-100 rounded-lg">
              <Activity className="w-6 h-6 text-purple-600" />
            </div>
          </div>
          <p className="text-xs text-gray-500">Features refreshed this week</p>
        </div>

        <div className={`rounded-lg shadow-sm border p-6 ${coverageBg}`}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-600">Cache Coverage</p>
              <p className={`text-3xl font-bold mt-2 ${coverageColor}`}>
                {cacheCoverage.toFixed(1)}%
              </p>
            </div>
            <div className="p-3 bg-white/50 rounded-lg">
              <Zap className="w-6 h-6 text-blue-600" />
            </div>
          </div>
          <p className="text-xs text-gray-500">Features available vs total predictions</p>
        </div>
      </div>

      {/* Feature Store Details */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Version Distribution */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Package className="w-5 h-5 text-blue-600" />
            Feature Version Distribution
          </h3>
          {stats.version_distribution && Object.keys(stats.version_distribution).length > 0 ? (
            <div className="space-y-3">
              {Object.entries(stats.version_distribution).map(([version, count]) => (
                <div key={version} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                    <span className="text-sm font-medium text-gray-700">{version || 'Unknown'}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ 
                          width: `${(count / stats.total_features) * 100}%` 
                        }}
                      />
                    </div>
                    <span className="text-sm font-semibold text-gray-900 w-16 text-right">
                      {count.toLocaleString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No version information available</p>
          )}
        </div>

        {/* Timestamps */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-600" />
            Feature Store Timeline
          </h3>
          <div className="space-y-4">
            <div>
              <p className="text-sm text-gray-600 mb-1">Oldest Feature</p>
              <p className="text-sm font-medium text-gray-900">
                {formatDate(stats.oldest_feature)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600 mb-1">Newest Feature</p>
              <p className="text-sm font-medium text-gray-900">
                {formatDate(stats.newest_feature)}
              </p>
            </div>
            {stats.oldest_feature && stats.newest_feature && (
              <div className="pt-4 border-t border-gray-200">
                <p className="text-sm text-gray-600 mb-1">Store Age</p>
                <p className="text-sm font-medium text-gray-900">
                  {Math.ceil(
                    (new Date(stats.newest_feature) - new Date(stats.oldest_feature)) / (1000 * 60 * 60 * 24)
                  )} days
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Performance Benefits */}
      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg border border-blue-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-blue-600" />
          Performance Benefits
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-yellow-500" />
              <span className="text-sm font-medium text-gray-700">Faster Predictions</span>
            </div>
            <p className="text-xs text-gray-600">
              Features retrieved from cache eliminate computation time, reducing prediction latency
            </p>
          </div>
          <div className="bg-white rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-green-500" />
              <span className="text-sm font-medium text-gray-700">Reduced Load</span>
            </div>
            <p className="text-xs text-gray-600">
              Pre-computed features reduce computational load on the feature engineering pipeline
            </p>
          </div>
          <div className="bg-white rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-4 h-4 text-blue-500" />
              <span className="text-sm font-medium text-gray-700">Consistency</span>
            </div>
            <p className="text-xs text-gray-600">
              Same feature values used across predictions ensure consistent model behavior
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeatureStore;
