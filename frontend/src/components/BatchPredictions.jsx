import React, { useState, useEffect } from 'react';
import { RefreshCw, Play, Clock, CheckCircle, XCircle, AlertTriangle, FileText, Database, Download, TrendingUp } from 'lucide-react';
import { creditScoringAPI } from '../utils/api';

const BatchPredictions = () => {
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadJobs();
  }, []);

  const loadJobs = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.listBatchPredictionJobs();
      setJobs(data.jobs || []);
    } catch (err) {
      setError(err.message || 'Failed to load batch prediction jobs');
      console.error('Error loading jobs:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadJobDetails = async (jobId) => {
    setLoading(true);
    setError(null);
    try {
      const data = await creditScoringAPI.getBatchPredictionJob(jobId);
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
      await creditScoringAPI.runBatchPredictionJob(jobId);
      await loadJobs();
      if (selectedJob && selectedJob.job_id === jobId) {
        await loadJobDetails(jobId);
      }
      alert('Batch prediction job started successfully');
    } catch (err) {
      setError(err.message || 'Failed to run batch prediction job');
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

  const formatFileSize = (bytes) => {
    if (!bytes) return 'N/A';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
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
      case 'paused':
        return 'text-gray-600 bg-gray-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  if (loading && jobs.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
        <p>Loading batch prediction jobs...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Batch Prediction Pipeline</h2>
          <p className="text-sm text-gray-600 mt-1">Large-scale prediction processing with multiple input/output formats</p>
        </div>
        <button
          onClick={loadJobs}
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

      {/* Jobs List and Details */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Jobs List */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Jobs</h3>
            <div className="space-y-2">
              {jobs.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-4">No batch prediction jobs found</p>
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
                    <div className="text-xs text-gray-600 space-y-1">
                      <div>Input: {job.input_source}</div>
                      <div>Output: {job.output_format}</div>
                      {job.total_records && (
                        <div>
                          Progress: {job.processed_records || 0}/{job.total_records} ({job.progress_percentage?.toFixed(1) || 0}%)
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
                    <p className="text-sm text-gray-600">Input Source</p>
                    <p className="text-sm font-medium text-gray-900">{selectedJob.input_source}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Output Format</p>
                    <p className="text-sm font-medium text-gray-900">{selectedJob.output_format}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Model</p>
                    <p className="text-sm font-medium text-gray-900">{selectedJob.model_name} ({selectedJob.model_stage})</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Model Version</p>
                    <p className="text-sm font-medium text-gray-900">{selectedJob.model_version || 'Latest'}</p>
                  </div>
                </div>

                {selectedJob.error_message && (
                  <div className="mt-4 p-3 bg-red-50 rounded-md">
                    <p className="text-sm text-red-700">{selectedJob.error_message}</p>
                  </div>
                )}
              </div>

              {/* Progress */}
              {selectedJob.total_records && (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Progress</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-600">Processing</span>
                        <span className="text-sm font-medium text-gray-900">
                          {selectedJob.processed_records || 0} / {selectedJob.total_records} records
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                          className="bg-blue-600 h-2.5 rounded-full transition-all"
                          style={{ width: `${selectedJob.progress_percentage || 0}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {selectedJob.progress_percentage?.toFixed(1) || 0}% complete
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className="text-xs text-gray-600">Total Records</p>
                        <p className="text-sm font-medium text-gray-900">{selectedJob.total_records || 0}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600">Processed</p>
                        <p className="text-sm font-medium text-green-600">{selectedJob.processed_records || 0}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600">Failed</p>
                        <p className="text-sm font-medium text-red-600">{selectedJob.failed_records || 0}</p>
                      </div>
                    </div>

                    {selectedJob.records_per_second && (
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <TrendingUp className="w-4 h-4" />
                        <span>{selectedJob.records_per_second.toFixed(2)} records/second</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Output */}
              {selectedJob.output_path && (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Output</h3>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Output Path</span>
                      <span className="text-sm font-medium text-gray-900 break-all">{selectedJob.output_path}</span>
                    </div>
                    {selectedJob.output_file_size_bytes && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">File Size</span>
                        <span className="text-sm font-medium text-gray-900">
                          {formatFileSize(selectedJob.output_file_size_bytes)}
                        </span>
                      </div>
                    )}
                    {selectedJob.output_path && selectedJob.output_format !== 'database' && (
                      <button
                        onClick={() => {
                          // In a real implementation, this would download the file
                          alert('Download functionality would be implemented here');
                        }}
                        className="mt-2 flex items-center gap-2 px-3 py-2 text-sm text-blue-700 bg-blue-50 rounded hover:bg-blue-100"
                      >
                        <Download className="w-4 h-4" />
                        Download Results
                      </button>
                    )}
                  </div>
                </div>
              )}

              {/* Timestamps */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Timestamps</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Created</span>
                    <span className="text-gray-900">{formatDate(selectedJob.created_at)}</span>
                  </div>
                  {selectedJob.started_at && (
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Started</span>
                      <span className="text-gray-900">{formatDate(selectedJob.started_at)}</span>
                    </div>
                  )}
                  {selectedJob.completed_at && (
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Completed</span>
                      <span className="text-gray-900">{formatDate(selectedJob.completed_at)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center text-gray-500">
              <FileText className="w-12 h-12 mx-auto mb-2 text-gray-400" />
              <p>Select a job to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BatchPredictions;
