import React, { useState, useEffect } from 'react';
import { RefreshCw, Play, TrendingUp, AlertTriangle, CheckCircle, XCircle, BarChart3, Zap, Clock } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const LoadTesting = () => {
  const [testResults, setTestResults] = useState(null);
  const [capacityEstimate, setCapacityEstimate] = useState(null);
  const [benchmarkResults, setBenchmarkResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('load-test');

  const [testConfig, setTestConfig] = useState({
    total_requests: 100,
    concurrent_users: 10,
    duration_seconds: null,
    scenarios: [
      {
        name: 'predict',
        endpoint: '/predict',
        method: 'POST',
        payload: { features: Array(26).fill(0), include_explanation: false },
        weight: 3
      },
      {
        name: 'health',
        endpoint: '/health',
        method: 'GET',
        payload: {},
        weight: 1
      }
    ]
  });

  const [capacityConfig, setCapacityConfig] = useState({
    target_rps: 100,
    avg_latency_ms: 150,
    target_p95_ms: 200
  });

  useEffect(() => {
    loadBenchmarkResults();
  }, []);

  const loadBenchmarkResults = async () => {
    try {
      const data = await creditScoringAPI.getBenchmarkResults();
      setBenchmarkResults(data);
    } catch (err) {
      console.error('Error loading benchmark results:', err);
    }
  };

  const handleRunLoadTest = async () => {
    setLoading(true);
    setError(null);
    try {
      const results = await creditScoringAPI.runLoadTest(testConfig);
      setTestResults(results);
    } catch (err) {
      setError(err.message || 'Failed to run load test');
      console.error('Error running load test:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleEstimateCapacity = async () => {
    setLoading(true);
    setError(null);
    try {
      const estimates = await creditScoringAPI.estimateCapacity(capacityConfig);
      setCapacityEstimate(estimates);
    } catch (err) {
      setError(err.message || 'Failed to estimate capacity');
      console.error('Error estimating capacity:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num) => {
    if (num === null || num === undefined) return 'N/A';
    return typeof num === 'number' ? num.toFixed(2) : num;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Load Testing & Performance Benchmarking</h2>
          <p className="text-sm text-gray-600 mt-1">Automated load testing, stress testing, and capacity planning</p>
        </div>
        <button
          onClick={loadBenchmarkResults}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
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

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('load-test')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'load-test'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Load Testing
          </button>
          <button
            onClick={() => setActiveTab('capacity')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'capacity'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Capacity Planning
          </button>
          <button
            onClick={() => setActiveTab('benchmark')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'benchmark'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Benchmarks
          </button>
        </nav>
      </div>

      {/* Load Testing Tab */}
      {activeTab === 'load-test' && (
        <div className="space-y-6">
          {/* Test Configuration */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Load Test Configuration</h3>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Total Requests
                </label>
                <input
                  type="number"
                  value={testConfig.total_requests}
                  onChange={(e) => setTestConfig({ ...testConfig, total_requests: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Concurrent Users
                </label>
                <input
                  type="number"
                  value={testConfig.concurrent_users}
                  onChange={(e) => setTestConfig({ ...testConfig, concurrent_users: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Duration (seconds, optional)
                </label>
                <input
                  type="number"
                  value={testConfig.duration_seconds || ''}
                  onChange={(e) => setTestConfig({ ...testConfig, duration_seconds: e.target.value ? parseInt(e.target.value) : null })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  placeholder="Optional"
                />
              </div>
            </div>
            <button
              onClick={handleRunLoadTest}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              Run Load Test
            </button>
          </div>

          {/* Test Results */}
          {testResults && (
            <div className="space-y-4">
              {/* Summary */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Test Results Summary</h3>
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Total Requests</p>
                    <p className="text-2xl font-bold text-gray-900">{testResults.total_requests}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Successful</p>
                    <p className="text-2xl font-bold text-green-600">{testResults.successful_requests}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Failed</p>
                    <p className="text-2xl font-bold text-red-600">{testResults.failed_requests}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Success Rate</p>
                    <p className="text-2xl font-bold text-gray-900">{formatNumber(testResults.success_rate)}%</p>
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-3 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Requests/Second</p>
                    <p className="text-lg font-medium text-gray-900">{formatNumber(testResults.requests_per_second)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Elapsed Time</p>
                    <p className="text-lg font-medium text-gray-900">{formatNumber(testResults.elapsed_time_seconds)}s</p>
                  </div>
                </div>
              </div>

              {/* Statistics */}
              {testResults.statistics && (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Latency Statistics</h3>
                  <div className="grid grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-gray-600">Mean</p>
                      <p className="text-lg font-medium text-gray-900">{formatNumber(testResults.statistics.mean_ms)}ms</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Median</p>
                      <p className="text-lg font-medium text-gray-900">{formatNumber(testResults.statistics.median_ms)}ms</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">P95</p>
                      <p className={`text-lg font-medium ${testResults.statistics.p95_ms > 200 ? 'text-red-600' : 'text-green-600'}`}>
                        {formatNumber(testResults.statistics.p95_ms)}ms
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">P99</p>
                      <p className="text-lg font-medium text-gray-900">{formatNumber(testResults.statistics.p99_ms)}ms</p>
                    </div>
                  </div>
                </div>
              )}

              {/* SLA Compliance */}
              {testResults.sla_compliance && (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">SLA Compliance</h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">P95 Latency Compliance</span>
                      {testResults.sla_compliance.p95_compliant ? (
                        <span className="flex items-center gap-1 text-sm text-green-600">
                          <CheckCircle className="w-4 h-4" />
                          Compliant
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-sm text-red-600">
                          <XCircle className="w-4 h-4" />
                          Non-Compliant
                        </span>
                      )}
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Success Rate Compliance</span>
                      {testResults.sla_compliance.success_rate_compliant ? (
                        <span className="flex items-center gap-1 text-sm text-green-600">
                          <CheckCircle className="w-4 h-4" />
                          Compliant
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-sm text-red-600">
                          <XCircle className="w-4 h-4" />
                          Non-Compliant
                        </span>
                      )}
                    </div>
                    <div className="flex items-center justify-between pt-2 border-t">
                      <span className="text-sm font-medium text-gray-900">Overall Compliance</span>
                      {testResults.sla_compliance.overall_compliant ? (
                        <span className="flex items-center gap-1 text-sm font-medium text-green-600">
                          <CheckCircle className="w-4 h-4" />
                          Compliant
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-sm font-medium text-red-600">
                          <XCircle className="w-4 h-4" />
                          Non-Compliant
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Capacity Planning Tab */}
      {activeTab === 'capacity' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Capacity Planning</h3>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Target RPS
                </label>
                <input
                  type="number"
                  value={capacityConfig.target_rps}
                  onChange={(e) => setCapacityConfig({ ...capacityConfig, target_rps: parseFloat(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Avg Latency (ms)
                </label>
                <input
                  type="number"
                  value={capacityConfig.avg_latency_ms}
                  onChange={(e) => setCapacityConfig({ ...capacityConfig, avg_latency_ms: parseFloat(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Target P95 (ms)
                </label>
                <input
                  type="number"
                  value={capacityConfig.target_p95_ms}
                  onChange={(e) => setCapacityConfig({ ...capacityConfig, target_p95_ms: parseFloat(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
            </div>
            <button
              onClick={handleEstimateCapacity}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              <TrendingUp className="w-4 h-4" />
              Estimate Capacity
            </button>
          </div>

          {capacityEstimate && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Capacity Estimates</h3>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                  <p className="text-sm text-gray-600">Concurrent Connections</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {formatNumber(capacityEstimate.estimated_concurrent_connections)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Workers Needed</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {formatNumber(capacityEstimate.estimated_workers)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Memory (MB)</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {formatNumber(capacityEstimate.estimated_memory_mb)}
                  </p>
                </div>
              </div>
              {capacityEstimate.recommendations && capacityEstimate.recommendations.length > 0 && (
                <div className="mt-4">
                  <p className="text-sm font-medium text-gray-900 mb-2">Recommendations:</p>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                    {capacityEstimate.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Benchmarks Tab */}
      {activeTab === 'benchmark' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Performance Benchmarks</h3>
          {benchmarkResults?.current_benchmark ? (
            <div className="space-y-4">
              {benchmarkResults.current_benchmark.statistics && (
                <div>
                  <p className="text-sm font-medium text-gray-900 mb-2">Statistics:</p>
                  <pre className="bg-gray-50 p-4 rounded-md text-sm overflow-auto">
                    {JSON.stringify(benchmarkResults.current_benchmark.statistics, null, 2)}
                  </pre>
                </div>
              )}
              {benchmarkResults.current_benchmark.sla_compliance && (
                <div>
                  <p className="text-sm font-medium text-gray-900 mb-2">SLA Compliance:</p>
                  <pre className="bg-gray-50 p-4 rounded-md text-sm overflow-auto">
                    {JSON.stringify(benchmarkResults.current_benchmark.sla_compliance, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No benchmark data available. Run a load test first.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default LoadTesting;
