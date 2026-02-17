import React, { useState, useEffect } from 'react';
import { FlaskConical, Play, Square, TrendingUp, BarChart3, CheckCircle, XCircle, Clock, AlertCircle, Plus, X } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';
import { useAuth } from '../contexts/AuthContext';

const ABTesting = () => {
  const { user, hasPermission } = useAuth();
  const [experiments, setExperiments] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [modelVersions, setModelVersions] = useState([]);
  const [createFormData, setCreateFormData] = useState({
    experiment_name: '',
    description: '',
    variants: [
      { name: 'control', model_version: 'Production', traffic_percentage: 50 },
      { name: 'treatment', model_version: 'Staging', traffic_percentage: 50 }
    ],
    traffic_percentage: 100,
    assignment_method: 'hash',
    primary_metric: 'accuracy',
    minimum_sample_size: 1000,
    significance_level: 0.05,
    minimum_improvement: 0.01
  });

  useEffect(() => {
    loadExperiments();
    loadModelVersions();
  }, []);

  const loadModelVersions = async () => {
    try {
      const data = await creditScoringAPI.getModelVersions();
      if (data.versions) {
        setModelVersions(data.versions);
      }
    } catch (err) {
      console.error('Error loading model versions:', err);
    }
  };

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

  const handleCreateExperiment = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      // Validate variants traffic percentages sum to 100
      const totalTraffic = createFormData.variants.reduce((sum, v) => sum + (v.traffic_percentage || 0), 0);
      if (totalTraffic !== 100) {
        setError(`Variant traffic percentages must sum to 100% (currently ${totalTraffic}%)`);
        setLoading(false);
        return;
      }

      await creditScoringAPI.createExperiment({
        experiment_name: createFormData.experiment_name,
        description: createFormData.description,
        variants: createFormData.variants,
        traffic_percentage: createFormData.traffic_percentage,
        assignment_method: createFormData.assignment_method,
        primary_metric: createFormData.primary_metric,
        minimum_sample_size: createFormData.minimum_sample_size,
        significance_level: createFormData.significance_level,
        minimum_improvement: createFormData.minimum_improvement
      });
      
      setShowCreateForm(false);
      setCreateFormData({
        experiment_name: '',
        description: '',
        variants: [
          { name: 'control', model_version: 'Production', traffic_percentage: 50 },
          { name: 'treatment', model_version: 'Staging', traffic_percentage: 50 }
        ],
        traffic_percentage: 100,
        assignment_method: 'hash',
        primary_metric: 'accuracy',
        minimum_sample_size: 1000,
        significance_level: 0.05,
        minimum_improvement: 0.01
      });
      loadExperiments();
    } catch (err) {
      // Extract error message from API response
      let errorMessage = 'Failed to create experiment';
      if (err.response) {
        if (err.response.status === 403) {
          errorMessage = 'Permission denied: You need "model:write" permission or superuser access to create experiments. Please contact your administrator.';
        } else if (err.response.data?.detail) {
          errorMessage = err.response.data.detail;
        } else if (err.response.status === 400) {
          errorMessage = err.response.data?.detail || 'Invalid experiment configuration. Please check your inputs.';
        }
      } else if (err.message) {
        errorMessage = err.message;
      }
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const addVariant = () => {
    setCreateFormData({
      ...createFormData,
      variants: [
        ...createFormData.variants,
        { name: `variant_${createFormData.variants.length + 1}`, model_version: 'Production', traffic_percentage: 0 }
      ]
    });
  };

  const removeVariant = (index) => {
    const newVariants = createFormData.variants.filter((_, i) => i !== index);
    // Redistribute traffic if removing a variant
    const removedTraffic = createFormData.variants[index].traffic_percentage;
    const remainingCount = newVariants.length;
    if (remainingCount > 0) {
      const perVariant = Math.floor(removedTraffic / remainingCount);
      const remainder = removedTraffic % remainingCount;
      newVariants.forEach((v, i) => {
        v.traffic_percentage = perVariant + (i < remainder ? 1 : 0);
      });
    }
    setCreateFormData({ ...createFormData, variants: newVariants });
  };

  const updateVariant = (index, field, value) => {
    const newVariants = [...createFormData.variants];
    newVariants[index] = { ...newVariants[index], [field]: value };
    setCreateFormData({ ...createFormData, variants: newVariants });
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
        <div className="flex items-center gap-2">
          {hasPermission('model:write') || user?.is_superuser ? (
            <button
              onClick={() => setShowCreateForm(!showCreateForm)}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              <Plus className="w-4 h-4" />
              Create Experiment
            </button>
          ) : (
            <div className="relative group">
              <div className="flex items-center gap-2 px-4 py-2 text-sm text-gray-500 bg-gray-100 rounded-md cursor-not-allowed">
                <Plus className="w-4 h-4" />
                Create Experiment
              </div>
              <div className="absolute left-0 top-full mt-2 w-64 p-3 bg-gray-800 text-white text-xs rounded-md shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                <p className="font-semibold mb-1">Permission Required</p>
                <p className="text-gray-300">
                  You need the <code className="bg-gray-700 px-1 rounded">model:write</code> permission or superuser access to create experiments.
                </p>
                <p className="text-gray-400 mt-2 text-xs">
                  Contact your administrator to assign the <code className="bg-gray-700 px-1 rounded">model_developer</code> role or run the migration script.
                </p>
              </div>
            </div>
          )}
          <button
            onClick={loadExperiments}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
          >
            <BarChart3 className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border-l-4 border-red-400">
          <div className="flex items-center gap-2 text-red-700">
            <XCircle className="w-5 h-5" />
            <span className="text-sm">{error}</span>
          </div>
        </div>
      )}

      {!hasPermission('model:write') && !user?.is_superuser && (
        <div className="p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded-md">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-yellow-800 mb-1">Permission Required</h3>
              <p className="text-sm text-yellow-700 mb-2">
                You need the <code className="bg-yellow-100 px-1.5 py-0.5 rounded text-xs font-mono">model:write</code> permission to create A/B testing experiments.
              </p>
              <div className="text-xs text-yellow-600 space-y-1">
                <p><strong>To fix this:</strong></p>
                <ol className="list-decimal list-inside ml-2 space-y-0.5">
                  <li>Run the migration script: <code className="bg-yellow-100 px-1 py-0.5 rounded">scripts/add_model_write_permission.sql</code></li>
                  <li>Or have an administrator assign you the <code className="bg-yellow-100 px-1 py-0.5 rounded">model_developer</code> role</li>
                  <li>Or log in as a superuser account</li>
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Create Experiment Form */}
      {showCreateForm && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Create New Experiment</h3>
            <button
              onClick={() => setShowCreateForm(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          
          <form onSubmit={handleCreateExperiment} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Experiment Name *
              </label>
              <input
                type="text"
                required
                value={createFormData.experiment_name}
                onChange={(e) => setCreateFormData({ ...createFormData, experiment_name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="e.g., Random Forest vs XGBoost"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={createFormData.description}
                onChange={(e) => setCreateFormData({ ...createFormData, description: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={2}
                placeholder="Describe the experiment purpose..."
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-sm font-medium text-gray-700">
                  Variants *
                </label>
                <button
                  type="button"
                  onClick={addVariant}
                  className="text-sm text-blue-600 hover:text-blue-700"
                >
                  + Add Variant
                </button>
              </div>
              <div className="space-y-3">
                {createFormData.variants.map((variant, index) => (
                  <div key={index} className="flex items-center gap-2 p-3 bg-gray-50 rounded-md">
                    <input
                      type="text"
                      required
                      value={variant.name}
                      onChange={(e) => updateVariant(index, 'name', e.target.value)}
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="Variant name"
                    />
                    <select
                      value={variant.model_version}
                      onChange={(e) => updateVariant(index, 'model_version', e.target.value)}
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="Production">Production</option>
                      <option value="Staging">Staging</option>
                      {modelVersions.map((mv) => (
                        <option key={mv.version} value={mv.version}>
                          Version {mv.version} ({mv.stage})
                        </option>
                      ))}
                    </select>
                    <input
                      type="number"
                      required
                      min="0"
                      max="100"
                      value={variant.traffic_percentage}
                      onChange={(e) => updateVariant(index, 'traffic_percentage', parseInt(e.target.value) || 0)}
                      className="w-24 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="%"
                    />
                    <span className="text-sm text-gray-600">%</span>
                    {createFormData.variants.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removeVariant(index)}
                        className="text-red-600 hover:text-red-700"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Total: {createFormData.variants.reduce((sum, v) => sum + (v.traffic_percentage || 0), 0)}%
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Primary Metric
                </label>
                <select
                  value={createFormData.primary_metric}
                  onChange={(e) => setCreateFormData({ ...createFormData, primary_metric: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="accuracy">Accuracy</option>
                  <option value="roc_auc">ROC-AUC</option>
                  <option value="precision">Precision</option>
                  <option value="recall">Recall</option>
                  <option value="f1">F1 Score</option>
                  <option value="latency">Latency</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Minimum Sample Size
                </label>
                <input
                  type="number"
                  required
                  min="100"
                  value={createFormData.minimum_sample_size}
                  onChange={(e) => setCreateFormData({ ...createFormData, minimum_sample_size: parseInt(e.target.value) || 1000 })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-4 border-t">
              <button
                type="button"
                onClick={() => setShowCreateForm(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                {loading ? 'Creating...' : 'Create Experiment'}
              </button>
            </div>
          </form>
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
