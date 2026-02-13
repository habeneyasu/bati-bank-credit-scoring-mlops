import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, TrendingDown, Clock, AlertCircle, CheckCircle, Zap, BarChart3 } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { creditScoringAPI } from '../utils/api';

const PerformanceMonitor = () => {
  const [performanceData, setPerformanceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5); // seconds

  useEffect(() => {
    loadPerformanceData();
    
    let interval;
    if (autoRefresh) {
      interval = setInterval(() => {
        loadPerformanceData();
      }, refreshInterval * 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, refreshInterval]);

  const loadPerformanceData = async () => {
    try {
      setLoading(true);
      // Add timeout to prevent hanging
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Request timeout')), 5000)
      );
      
      const dataPromise = creditScoringAPI.getPerformanceMetrics();
      const data = await Promise.race([dataPromise, timeoutPromise]);
      
      setPerformanceData(data);
    } catch (error) {
      console.error('Failed to load performance data:', error);
      // Set error state so user knows what happened
      setPerformanceData({
        status: 'error',
        message: error.message || 'Failed to load performance metrics',
        stats: {},
        sla: { compliant: false, message: error.message }
      });
    } finally {
      setLoading(false);
    }
  };

  if (loading && !performanceData) {
    return (
      <div className="card animate-fade-in">
        <div className="text-center py-12">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-600">Loading performance metrics...</p>
          <p className="text-xs text-slate-500 mt-2">This should only take a moment</p>
        </div>
      </div>
    );
  }

  if (!performanceData) {
    return (
      <div className="card animate-fade-in">
        <div className="text-center py-12">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-slate-600">Failed to load performance data</p>
          <button
            onClick={loadPerformanceData}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Handle disabled status
  if (performanceData.status === 'disabled') {
    return (
      <div className="card animate-fade-in">
        <div className="text-center py-12">
          <AlertCircle className="w-12 h-12 text-amber-500 mx-auto mb-4" />
          <p className="text-slate-600 font-semibold mb-2">Performance monitoring is disabled</p>
          <p className="text-sm text-slate-500">{performanceData.message || 'Enable performance monitoring in API settings'}</p>
        </div>
      </div>
    );
  }

  // Handle error status
  if (performanceData.status === 'error' || !performanceData.stats) {
    return (
      <div className="card animate-fade-in">
        <div className="text-center py-12">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-slate-600 font-semibold mb-2">Error loading performance metrics</p>
          <p className="text-sm text-slate-500 mb-4">{performanceData.message || 'Unknown error occurred'}</p>
          <button
            onClick={loadPerformanceData}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const allStats = performanceData.stats?.all || {};
  const predictStats = performanceData.stats?.predict || {};
  const sla = performanceData.sla || {};
  const targetP95 = performanceData.target_p95_ms || 200;

  // Show empty state if no data collected yet
  if (allStats.count === 0) {
    return (
      <div className="card animate-fade-in">
        <div className="text-center py-12">
          <Activity className="w-12 h-12 text-slate-300 mx-auto mb-4" />
          <p className="text-slate-600 font-semibold mb-2">No Performance Data Yet</p>
          <p className="text-sm text-slate-500 mb-4">
            Performance metrics will appear here after making API requests.
            <br />
            Try making a prediction to see latency metrics.
          </p>
          <button
            onClick={loadPerformanceData}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>
    );
  }

  const formatMs = (value) => {
    if (value === undefined || value === null) return 'N/A';
    return `${value.toFixed(2)}ms`;
  };

  const getSlaStatus = () => {
    if (!sla.compliant) {
      return {
        icon: AlertCircle,
        color: 'red',
        text: 'Non-Compliant',
        bgColor: 'bg-red-100',
        textColor: 'text-red-700'
      };
    }
    return {
      icon: CheckCircle,
      color: 'green',
      text: 'Compliant',
      bgColor: 'bg-green-100',
      textColor: 'text-green-700'
    };
  };

  const slaStatus = getSlaStatus();
  const StatusIcon = slaStatus.icon;

  // Prepare chart data
  const latencyData = [
    { metric: 'Mean', value: allStats.mean || 0, target: targetP95 },
    { metric: 'Median', value: allStats.median || 0, target: targetP95 },
    { metric: 'P50', value: allStats.p50 || 0, target: targetP95 },
    { metric: 'P95', value: allStats.p95 || 0, target: targetP95 },
    { metric: 'P99', value: allStats.p99 || 0, target: targetP95 },
  ];

  return (
    <div className="card animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-100 rounded-lg">
            <Activity className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-slate-800">Performance Monitor</h2>
            <p className="text-slate-600 text-sm">Real-time API performance metrics and SLA compliance</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-slate-600">Auto-refresh:</label>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-4 h-4"
            />
          </div>
          {autoRefresh && (
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="px-3 py-1 border border-slate-300 rounded-lg text-sm"
            >
              <option value={5}>5s</option>
              <option value={10}>10s</option>
              <option value={30}>30s</option>
              <option value={60}>1m</option>
            </select>
          )}
          <button
            onClick={loadPerformanceData}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <Zap className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* SLA Status */}
      <div className={`rounded-lg p-6 mb-6 border-2 ${slaStatus.bgColor} border-${slaStatus.color}-300`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <StatusIcon className={`w-8 h-8 text-${slaStatus.color}-600`} />
            <div>
              <h3 className="text-lg font-bold text-slate-800">SLA Compliance Status</h3>
              <p className="text-sm text-slate-600">95th Percentile Latency Target: {targetP95}ms</p>
            </div>
          </div>
          <div className="text-right">
            <div className={`px-4 py-2 rounded-lg font-bold text-lg ${slaStatus.bgColor} ${slaStatus.textColor}`}>
              {slaStatus.text}
            </div>
            {sla.p95_ms && (
              <div className="text-sm text-slate-600 mt-2">
                Current P95: {formatMs(sla.p95_ms)}
                {sla.margin_ms && (
                  <span className={`ml-2 ${sla.margin_ms > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    ({sla.margin_ms > 0 ? '+' : ''}{formatMs(sla.margin_ms)} margin)
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-5 h-5 text-blue-600" />
            <div className="text-xs text-slate-600 uppercase tracking-wide">Mean Latency</div>
          </div>
          <div className="text-2xl font-bold text-slate-800">{formatMs(allStats.mean)}</div>
          <div className="text-xs text-slate-500 mt-1">Average response time</div>
        </div>

        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="flex items-center gap-2 mb-2">
            <BarChart3 className="w-5 h-5 text-purple-600" />
            <div className="text-xs text-slate-600 uppercase tracking-wide">P95 Latency</div>
          </div>
          <div className={`text-2xl font-bold ${(allStats.p95 || 0) <= targetP95 ? 'text-green-600' : 'text-red-600'}`}>
            {formatMs(allStats.p95)}
          </div>
          <div className="text-xs text-slate-500 mt-1">95th percentile</div>
        </div>

        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-5 h-5 text-orange-600" />
            <div className="text-xs text-slate-600 uppercase tracking-wide">P99 Latency</div>
          </div>
          <div className="text-2xl font-bold text-slate-800">{formatMs(allStats.p99)}</div>
          <div className="text-xs text-slate-500 mt-1">99th percentile</div>
        </div>

        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-5 h-5 text-green-600" />
            <div className="text-xs text-slate-600 uppercase tracking-wide">Total Requests</div>
          </div>
          <div className="text-2xl font-bold text-slate-800">{allStats.count || 0}</div>
          <div className="text-xs text-slate-500 mt-1">Samples collected</div>
        </div>
      </div>

      {/* Latency Chart */}
      <div className="bg-slate-50 rounded-lg p-6 border border-slate-200 mb-6">
        <h3 className="text-lg font-bold text-slate-800 mb-4">Latency Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={latencyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="metric"
              tick={{ fill: '#64748b', fontSize: 12 }}
              stroke="#cbd5e1"
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 12 }}
              stroke="#cbd5e1"
              label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              formatter={(value) => `${Number(value).toFixed(2)}ms`}
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Bar dataKey="value" name="Actual" fill="#3b82f6" radius={[8, 8, 0, 0]} />
            <Bar dataKey="target" name="Target (P95)" fill="#ef4444" radius={[8, 8, 0, 0]} opacity={0.5} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg p-6 border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-4">All Endpoints</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-slate-600">Mean:</span>
              <span className="font-semibold text-slate-800">{formatMs(allStats.mean)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Median:</span>
              <span className="font-semibold text-slate-800">{formatMs(allStats.median)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">P50:</span>
              <span className="font-semibold text-slate-800">{formatMs(allStats.p50)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">P95:</span>
              <span className={`font-semibold ${(allStats.p95 || 0) <= targetP95 ? 'text-green-600' : 'text-red-600'}`}>
                {formatMs(allStats.p95)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">P99:</span>
              <span className="font-semibold text-slate-800">{formatMs(allStats.p99)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Min:</span>
              <span className="font-semibold text-slate-800">{formatMs(allStats.min)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Max:</span>
              <span className="font-semibold text-slate-800">{formatMs(allStats.max)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Std Dev:</span>
              <span className="font-semibold text-slate-800">{formatMs(allStats.std)}</span>
            </div>
          </div>
        </div>

        {predictStats && Object.keys(predictStats).length > 0 && (
          <div className="bg-white rounded-lg p-6 border border-slate-200">
            <h3 className="text-lg font-bold text-slate-800 mb-4">/predict Endpoint</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-slate-600">Mean:</span>
                <span className="font-semibold text-slate-800">{formatMs(predictStats.mean)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">P95:</span>
                <span className={`font-semibold ${(predictStats.p95 || 0) <= targetP95 ? 'text-green-600' : 'text-red-600'}`}>
                  {formatMs(predictStats.p95)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">P99:</span>
                <span className="font-semibold text-slate-800">{formatMs(predictStats.p99)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">Requests:</span>
                <span className="font-semibold text-slate-800">{predictStats.count || 0}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error Rate */}
      {performanceData.stats.error_rate && (
        <div className="mt-6 bg-amber-50 rounded-lg p-6 border border-amber-200">
          <h3 className="text-lg font-bold text-slate-800 mb-4">Error Rate</h3>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold text-slate-800">
                {(performanceData.stats.error_rate.rate * 100).toFixed(2)}%
              </div>
              <div className="text-sm text-slate-600">
                {performanceData.stats.error_rate.errors} errors out of {performanceData.stats.error_rate.total} requests
              </div>
            </div>
            {performanceData.stats.error_rate.rate > 0.01 && (
              <AlertCircle className="w-8 h-8 text-amber-600" />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceMonitor;
