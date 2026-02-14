import React, { useState, useEffect } from 'react';
import { FlaskConical, Play, Square, TrendingUp, BarChart3, CheckCircle, XCircle, Clock, AlertCircle } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const ABTesting = () => {
  const [experiments, setExperiments] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadExperiments();
  }, []);

  useEffect(() => {
    if (selectedExperiment) {
      loadExperimentResults(selectedExperiment);
    }
  }, [selectedExperiment]);

  const loadExperiments = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.listExperiments();
      setExperiments(data.experiments || []);
    } catch (err) {
      setError(err.message || 'Failed to load experiments');
      console.error('Error loading experiments:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadExperimentResults = async (experimentId) => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.getExperimentResults(experimentId);
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed to load experiment results');
      console.error('Error loading experiment results:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleStartExperiment = async (experimentId) => {
    try {
      await creditScoringAPI.startExperiment(experimentId);
      loadExperiments();
      if (selectedExperiment === experimentId) {
        loadExperimentResults(experimentId);
      }
    } catch (err) {
      setError(err.message || 'Failed to start experiment');
    }
  };

  const handleStopExperiment = async (experimentId) => {
    try {
      await creditScoringAPI.stopExperiment(experimentId);
      loadExperiments();
      if (selectedExperiment === experimentId) {
        loadExperimentResults(experimentId);
      }
    } catch (err) {
      setError(err.message || 'Failed to stop experiment');
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

  const getStatusColor = (status) => {
    switch (status) {
      case 'running':
        return 'text-green-600 bg-green-50';
      case 'completed':
        return 'text-blue-600 bg-blue-50';
      case 'paused':
        return 'text-yellow-600 bg-yellow-50';
      case 'draft':
        return 'text-gray-600 bg-gray-50';
      case 'cancelled':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  if (loading && experiments.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <FlaskConical className="w-8 h-8 animate-pulse mx-auto mb-2" />
        <p>Loading experiments...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">A/B Testing Experiments</h2>
          <p className="text-sm text-gray-600 mt-1">Compare model variants and determine winners</p>
        </div>
        <button
          onClick={loadExperiments}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
        >
          <BarChart3 className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border-l-4 border-red-400">
          <div className="flex items-center gap-2 text-red-700">
            <XCircle className="w-5 h-5" />
            <span className="text-sm">{error}</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Experiments List */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Experiments</h3>
            <div className="space-y-2">
              {experiments.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-4">No experiments found</p>
              ) : (
                experiments.map((exp) => (
                  <div
                    key={exp.experiment_id}
                    onClick={() => setSelectedExperiment(exp.experiment_id)}
                    className={`p-3 rounded-md cursor-pointer border transition-colors ${
                      selectedExperiment === exp.experiment_id
                        ? 'bg-blue-50 border-blue-300'
                        : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-sm text-gray-900">{exp.experiment_name}</span>
                      <span className={`px-2 py-1 text-xs font-medium rounded ${getStatusColor(exp.status)}`}>
                        {exp.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                      {exp.status === 'draft' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStartExperiment(exp.experiment_id);
                          }}
                          className="flex items-center gap-1 px-2 py-1 text-xs text-green-700 bg-green-50 rounded hover:bg-green-100"
                        >
                          <Play className="w-3 h-3" />
                          Start
                        </button>
                      )}
                      {exp.status === 'running' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStopExperiment(exp.experiment_id);
                          }}
                          className="flex items-center gap-1 px-2 py-1 text-xs text-red-700 bg-red-50 rounded hover:bg-red-100"
                        >
                          <Square className="w-3 h-3" />
                          Stop
                        </button>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Experiment Details */}
        <div className="lg:col-span-2">
          {selectedExperiment && results ? (
            <div className="space-y-4">
              {/* Experiment Info */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">{results.experiment_name}</h3>
                  <span className={`px-3 py-1 text-sm font-medium rounded ${getStatusColor(results.status)}`}>
                    {results.status}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-sm text-gray-600">Primary Metric</p>
                    <p className="text-sm font-medium text-gray-900">{results.primary_metric || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Winner</p>
                    <p className="text-sm font-medium text-gray-900">
                      {results.winner?.winner || results.winner_variant || 'TBD'}
                    </p>
                  </div>
                </div>

                {results.conclusion && (
                  <div className="mt-4 p-3 bg-blue-50 rounded-md">
                    <p className="text-sm text-gray-700">{results.conclusion}</p>
                  </div>
                )}
              </div>

              {/* Variant Metrics */}
              {results.variant_metrics && Object.keys(results.variant_metrics).length > 0 && (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Variant Performance</h3>
                  <div className="space-y-4">
                    {Object.entries(results.variant_metrics).map(([variantName, metrics]) => (
                      <div key={variantName} className="border border-gray-200 rounded-md p-4">
                        <div className="flex items-center justify-between mb-3">
                          <h4 className="font-medium text-gray-900">{variantName}</h4>
                          {results.winner?.winner === variantName && (
                            <span className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-green-700 bg-green-50 rounded">
                              <CheckCircle className="w-3 h-3" />
                              Winner
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div>
                            <p className="text-xs text-gray-600">Sample Size</p>
                            <p className="text-sm font-medium text-gray-900">
                              {metrics.sample_size?.toLocaleString() || 0}
                            </p>
                          </div>
                          {metrics.accuracy !== null && metrics.accuracy !== undefined && (
                            <div>
                              <p className="text-xs text-gray-600">Accuracy</p>
                              <p className="text-sm font-medium text-gray-900">
                                {(metrics.accuracy * 100).toFixed(2)}%
                              </p>
                            </div>
                          )}
                          {metrics.mean_probability !== null && metrics.mean_probability !== undefined && (
                            <div>
                              <p className="text-xs text-gray-600">Mean Probability</p>
                              <p className="text-sm font-medium text-gray-900">
                                {(metrics.mean_probability * 100).toFixed(2)}%
                              </p>
                            </div>
                          )}
                          {metrics.avg_latency_ms && (
                            <div>
                              <p className="text-xs text-gray-600">Avg Latency</p>
                              <p className="text-sm font-medium text-gray-900">
                                {metrics.avg_latency_ms.toFixed(2)}ms
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Statistical Analysis */}
              {results.winner && (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Statistical Analysis</h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Winner Variant</span>
                      <span className="text-sm font-medium text-gray-900">{results.winner.winner}</span>
                    </div>
                    {results.winner.improvement_pct !== undefined && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Improvement</span>
                        <span className="text-sm font-medium text-gray-900">
                          {results.winner.improvement_pct > 0 ? '+' : ''}
                          {results.winner.improvement_pct.toFixed(2)}%
                        </span>
                      </div>
                    )}
                    {results.statistical_significance !== null && results.statistical_significance !== undefined && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">P-Value</span>
                        <span className="text-sm font-medium text-gray-900">
                          {results.statistical_significance.toFixed(4)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : selectedExperiment ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center text-gray-500">
              <Clock className="w-12 h-12 mx-auto mb-2 text-gray-400" />
              <p>Loading experiment details...</p>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center text-gray-500">
              <FlaskConical className="w-12 h-12 mx-auto mb-2 text-gray-400" />
              <p>Select an experiment to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ABTesting;
