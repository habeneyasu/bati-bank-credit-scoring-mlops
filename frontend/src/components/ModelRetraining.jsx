import React, { useState, useEffect } from 'react';
import { RefreshCw, Play, Clock, CheckCircle, XCircle, AlertTriangle, TrendingUp, Settings, Calendar, Zap } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const ModelRetraining = () => {
  const [jobs, setJobs] = useState([]);
  const [schedules, setSchedules] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('jobs');

  useEffect(() => {
    loadJobs();
    loadSchedules();
  }, []);

  const loadJobs = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.listRetrainingJobs();
      setJobs(data.jobs || []);
    } catch (err) {
      setError(err.message || 'Failed to load retraining jobs');
      console.error('Error loading jobs:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadSchedules = async () => {
    try {
      const data = await creditScoringAPI.listRetrainingSchedules();
      setSchedules(data.schedules || []);
    } catch (err) {
      console.error('Error loading schedules:', err);
    }
  };

  const loadJobDetails = async (jobId) => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.getRetrainingJob(jobId);
      setSelectedJob(data);
    } catch (err) {
      setError(err.message || 'Failed to load job details');
      console.error('Error loading job details:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRunJob = async (jobId) => {
    try {
      setLoading(true);
      await creditScoringAPI.runRetrainingJob(jobId);
      await loadJobs();
      if (selectedJob && selectedJob.job_id === jobId) {
        await loadJobDetails(jobId);
      }
      alert('Retraining job started successfully');
    } catch (err) {
      setError(err.message || 'Failed to run retraining job');
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

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-50';
      case 'running':
        return 'text-blue-600 bg-blue-50';
      case 'failed':
        return 'text-red-600 bg-red-50';
      case 'pending':
        return 'text-yellow-600 bg-yellow-50';
      case 'cancelled':
        return 'text-gray-600 bg-gray-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const getPromotionStatusColor = (status) => {
    switch (status) {
      case 'promoted':
        return 'text-green-600 bg-green-50';
      case 'rejected':
        return 'text-red-600 bg-red-50';
      case 'rolled_back':
        return 'text-orange-600 bg-orange-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  if (loading && jobs.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
        <p>Loading retraining jobs...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Model Retraining Pipeline</h2>
          <p className="text-sm text-gray-600 mt-1">Automated model retraining, validation, and promotion</p>
        </div>
        <button
          onClick={() => { loadJobs(); loadSchedules(); }}
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
            onClick={() => setActiveTab('jobs')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'jobs'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Retraining Jobs
          </button>
          <button
            onClick={() => setActiveTab('schedules')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'schedules'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Schedules
          </button>
        </nav>
      </div>

      {/* Jobs Tab */}
      {activeTab === 'jobs' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Jobs List */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Jobs</h3>
              <div className="space-y-2">
                {jobs.length === 0 ? (
                  <p className="text-sm text-gray-500 text-center py-4">No retraining jobs found</p>
                ) : (
                  jobs.map((job) => (
                    <div
                      key={job.job_id}
                      onClick={() => loadJobDetails(job.job_id)}
                      className={`p-3 rounded-md cursor-pointer border transition-colors ${
                        selectedJob?.job_id === job.job_id
                          ? 'bg-blue-50 border-blue-300'
                          : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-sm text-gray-900">{job.job_name}</span>
                        <span className={`px-2 py-1 text-xs font-medium rounded ${getStatusColor(job.status)}`}>
                          {job.status}
                        </span>
                      </div>
                      <div className="text-xs text-gray-600">
                        <div>Model: {job.model_name}</div>
                        <div>Trigger: {job.trigger_type}</div>
                        {job.promotion_status && (
                          <div className="mt-1">
                            <span className={`px-2 py-1 text-xs font-medium rounded ${getPromotionStatusColor(job.promotion_status)}`}>
                              {job.promotion_status}
                            </span>
                          </div>
                        )}
                      </div>
                      {job.status === 'pending' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRunJob(job.job_id);
                          }}
                          className="mt-2 flex items-center gap-1 px-2 py-1 text-xs text-green-700 bg-green-50 rounded hover:bg-green-100"
                        >
                          <Play className="w-3 h-3" />
                          Run
                        </button>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Job Details */}
          <div className="lg:col-span-2">
            {selectedJob ? (
              <div className="space-y-4">
                {/* Job Info */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">{selectedJob.job_name}</h3>
                    <span className={`px-3 py-1 text-sm font-medium rounded ${getStatusColor(selectedJob.status)}`}>
                      {selectedJob.status}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-sm text-gray-600">Model</p>
                      <p className="text-sm font-medium text-gray-900">{selectedJob.model_name}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Model Type</p>
                      <p className="text-sm font-medium text-gray-900">{selectedJob.model_type || 'N/A'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Trigger</p>
                      <p className="text-sm font-medium text-gray-900">{selectedJob.trigger_type}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Model Version</p>
                      <p className="text-sm font-medium text-gray-900">{selectedJob.model_version || 'N/A'}</p>
                    </div>
                  </div>

                  {selectedJob.error_message && (
                    <div className="mt-4 p-3 bg-red-50 rounded-md">
                      <p className="text-sm text-red-700">{selectedJob.error_message}</p>
                    </div>
                  )}
                </div>

                {/* Validation Results */}
                {selectedJob.validation_passed !== null && (
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Validation Results</h3>
                    <div className="flex items-center gap-2 mb-4">
                      {selectedJob.validation_passed ? (
                        <span className="flex items-center gap-1 px-3 py-1 text-sm font-medium text-green-700 bg-green-50 rounded">
                          <CheckCircle className="w-4 h-4" />
                          Validation Passed
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 px-3 py-1 text-sm font-medium text-red-700 bg-red-50 rounded">
                          <XCircle className="w-4 h-4" />
                          Validation Failed
                        </span>
                      )}
                    </div>
                    {selectedJob.validation_errors && selectedJob.validation_errors.length > 0 && (
                      <div className="mt-4">
                        <p className="text-sm font-medium text-gray-900 mb-2">Errors:</p>
                        <ul className="list-disc list-inside text-sm text-red-700">
                          {selectedJob.validation_errors.map((error, idx) => (
                            <li key={idx}>{error}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {/* Metrics */}
                {selectedJob.test_metrics && (
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Metrics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(selectedJob.test_metrics).map(([metric, value]) => (
                        <div key={metric}>
                          <p className="text-xs text-gray-600 capitalize">{metric.replace('_', ' ')}</p>
                          <p className="text-sm font-medium text-gray-900">
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Baseline Comparison */}
                {selectedJob.baseline_comparison && (
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Baseline Comparison</h3>
                    {selectedJob.baseline_comparison.improvement && (
                      <div className="space-y-2">
                        {Object.entries(selectedJob.baseline_comparison.improvement).map(([metric, improvement]) => (
                          <div key={metric} className="flex items-center justify-between">
                            <span className="text-sm text-gray-600 capitalize">{metric.replace('_', ' ')}</span>
                            <span className={`text-sm font-medium ${improvement >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {improvement >= 0 ? '+' : ''}{improvement.toFixed(2)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Promotion Status */}
                {selectedJob.promotion_status && (
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Promotion Status</h3>
                    <div className="flex items-center gap-2">
                      <span className={`px-3 py-1 text-sm font-medium rounded ${getPromotionStatusColor(selectedJob.promotion_status)}`}>
                        {selectedJob.promotion_status}
                      </span>
                      {selectedJob.promoted_to_stage && (
                        <span className="text-sm text-gray-600">→ {selectedJob.promoted_to_stage}</span>
                      )}
                    </div>
                    {selectedJob.promotion_timestamp && (
                      <p className="text-xs text-gray-500 mt-2">
                        Promoted at: {formatDate(selectedJob.promotion_timestamp)}
                      </p>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center text-gray-500">
                <RefreshCw className="w-12 h-12 mx-auto mb-2 text-gray-400" />
                <p>Select a job to view details</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Schedules Tab */}
      {activeTab === 'schedules' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Retraining Schedules</h3>
          {schedules.length === 0 ? (
            <p className="text-sm text-gray-500 text-center py-4">No schedules configured</p>
          ) : (
            <div className="space-y-4">
              {schedules.map((schedule) => (
                <div key={schedule.schedule_id} className="border border-gray-200 rounded-md p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-900">{schedule.schedule_name}</span>
                    <span className={`px-2 py-1 text-xs font-medium rounded ${
                      schedule.is_active ? 'text-green-600 bg-green-50' : 'text-gray-600 bg-gray-50'
                    }`}>
                      {schedule.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600 space-y-1">
                    <div>Model: {schedule.model_name}</div>
                    <div>Schedule: {schedule.schedule_type}</div>
                    {schedule.next_run_at && (
                      <div>Next Run: {formatDate(schedule.next_run_at)}</div>
                    )}
                    {schedule.last_run_at && (
                      <div>Last Run: {formatDate(schedule.last_run_at)}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelRetraining;
